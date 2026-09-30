import pathlib
import socket
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'engine'))
from launch import wait_for_port_release


class PortReleaseTests(unittest.TestCase):
    def test_occupied_port_is_not_reported_released_and_can_be_reused_after_close(self):
        with socket.socket() as owner:
            owner.bind(('127.0.0.1', 0))
            port = owner.getsockname()[1]
            owner.listen(1)
            with self.assertRaisesRegex(TimeoutError, str(port)):
                wait_for_port_release([port], timeout=0)
        wait_for_port_release([port], timeout=1)
        with socket.socket() as consumer:
            consumer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            consumer.bind(('127.0.0.1', port))
            consumer.listen(1)
            with socket.create_connection(('127.0.0.1', port)) as client:
                with consumer.accept()[0] as server:
                    server.sendall(b'restarted')
                    self.assertEqual(client.recv(9), b'restarted')


if __name__ == '__main__':
    unittest.main()
