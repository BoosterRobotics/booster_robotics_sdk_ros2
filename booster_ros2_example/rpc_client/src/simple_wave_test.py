import rclpy
from rclpy.node import Node
from booster_interface.srv import RpcService
from booster_interface.msg import BoosterApiReqMsg
import json


def create_msg(api_id, param_dict=None):
    msg = BoosterApiReqMsg()
    msg.api_id = api_id
    if param_dict is not None:
        msg.body = json.dumps(param_dict)
    else:
        msg.body = ""
    return msg


def main():
    rclpy.init()
    node = Node('simple_wave_test_node')
    client = node.create_client(RpcService, 'booster_rpc_service')

    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')
        if not rclpy.ok():
            node.get_logger().error('Interrupted while waiting for the service. Exiting.')
            return

    node.get_logger().info('Sending wavehand (api_id 2005)...')

    # WaveHand requires parameters (no angles, but still needs hand_index/hand_action).
    # From Booster SDK headers:
    #   HandIndex: kLeftHand=0, kRightHand=1
    #   HandAction: kHandOpen=0, kHandClose=1
    # The SDK client uses right hand by default.
    req_wave = create_msg(2005, {"hand_index": 0, "hand_action": 0})
    request = RpcService.Request()
    request.msg = req_wave

    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is not None:
        resp = future.result().msg
        node.get_logger().info('Wavehand status: %s' % resp.status)
        node.get_logger().info('Wavehand body: %s' % resp.body)
    else:
        node.get_logger().error('Failed to call rpc service')

    node.get_logger().info('Wave test completed!')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
