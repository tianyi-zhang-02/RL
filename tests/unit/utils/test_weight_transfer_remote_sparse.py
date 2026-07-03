# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch
import zstandard

from nemo_rl.utils import weight_transfer_remote_sparse
from nemo_rl.utils.weight_transfer_remote_sparse import download_s3_refit_payload
from nemo_rl.utils.weight_transfer_sparse_codec import (
    DeltaCompressionTracker,
    encode_sparse_infos,
    sparse_locations_for_item,
)
from nemo_rl.utils.weight_transfer_zmq import (
    G_VLLM_REFIT_CHECKSUM_HEADER,
    G_VLLM_REFIT_PAYLOAD_HEADER,
    G_VLLM_REFIT_PRODUCER_HEADER,
    G_VLLM_REFIT_TRANSFER_HEADER,
    ZmqSparseRefitClient,
    ZmqSparseRefitServer,
    sparse_payload_checksum,
)


def test_delta_tracker_commits_only_successful_syncs(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = DeltaCompressionTracker(
        {"dtype": "bf16", "sparse_bucket_size_bytes": 1024}
    )
    tensor = torch.tensor([1.0, 2.0, 3.0])
    tracker.snapshot_baseline([("weight", tensor)])
    tensor[1] += 4

    assert tracker.prepare_sparse_delta_payload([("weight", tensor)])[2]
    tracker.on_sync_failed()
    assert tracker.prepare_sparse_delta_payload([("weight", tensor)])[2]
    tracker.on_sync_succeeded()
    assert not tracker.prepare_sparse_delta_payload([("weight", tensor)])[2]


def test_sparse_index_encoding_preserves_uint64_locations() -> None:
    locations = torch.tensor([0, 2**32 + 5])
    packed, _, metadata = encode_sparse_infos(
        [("weight", torch.empty(2), locations, torch.ones(2))],
        empty_dtype=torch.float32,
    )

    decoded = sparse_locations_for_item(metadata[0], packed, device="cpu")
    assert torch.equal(decoded, locations)


def test_s3_download_verifies_checksum(monkeypatch) -> None:
    compressed = zstandard.ZstdCompressor().compress(b"payload")
    monkeypatch.setattr(
        weight_transfer_remote_sparse,
        "_get_manifest_s3_store",
        lambda *_args: SimpleNamespace(get=lambda _key: compressed),
    )
    manifest = {
        "bucket": "bucket",
        "region": "region",
        "key": "key",
        "checksum": sparse_payload_checksum(compressed),
    }

    assert download_s3_refit_payload(manifest) == b"payload"
    manifest["checksum"] = "0" * 32
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_s3_refit_payload(manifest)


def _receiver_server(received):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            received.append(
                ({key.lower(): value for key, value in self.headers.items()}, body)
            )
            response = json.dumps({"ok": True, "receiver_total_s": 0.25}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _send_zmq_payload(
    client: ZmqSparseRefitClient,
    payload_id: int,
    body: bytes,
    checksum: str | None = None,
) -> dict[str, object]:
    return client.send_payload(
        transfer_id="transfer-a",
        payload_id=payload_id,
        checksum=checksum or sparse_payload_checksum(body),
        body=body,
    )


def test_zmq_sparse_refit_relay_fans_out_and_rejects_corruption(monkeypatch) -> None:
    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")
    received = [[], []]
    receivers = [_receiver_server(items) for items in received]
    urls = [f"http://127.0.0.1:{server.server_port}" for server, _ in receivers]
    relay = ZmqSparseRefitServer(
        urls,
        bind_address="tcp://127.0.0.1:*",
        api_key_env_var="NRL_TEST_REFIT_KEY",
        timeout_s=5.0,
    )
    address = relay.start()
    unauthenticated_client = ZmqSparseRefitClient(
        address,
        timeout_s=5.0,
        producer_id=2,
    )
    client = ZmqSparseRefitClient(
        address,
        timeout_s=5.0,
        producer_id=3,
        api_key="secret",
    )
    body = b"compressed sparse payload"
    checksum = sparse_payload_checksum(body)
    try:
        with pytest.raises(RuntimeError, match="authentication failed"):
            _send_zmq_payload(unauthenticated_client, 6, body)
        first = _send_zmq_payload(client, 7, body)
        assert first["ok"]
        assert first["receiver_total_s"] == 0.25
        assert [len(items) for items in received] == [1, 1]
        for items in received:
            headers, posted_body = items[0]
            assert posted_body == body
            assert headers[G_VLLM_REFIT_TRANSFER_HEADER] == "transfer-a"
            assert headers[G_VLLM_REFIT_PRODUCER_HEADER] == "3"
            assert headers[G_VLLM_REFIT_PAYLOAD_HEADER] == "7"
            assert headers[G_VLLM_REFIT_CHECKSUM_HEADER] == checksum
            assert headers["x-nemo-rl-refit-key"] == "secret"

        with pytest.raises(RuntimeError, match="checksum mismatch"):
            _send_zmq_payload(client, 8, body, "0" * 32)
    finally:
        unauthenticated_client.close()
        client.close()
        relay.close()
        for server, thread in receivers:
            server.shutdown()
            thread.join(timeout=5.0)
            server.server_close()
