import rclpy
from rclpy.node import Node
from booster_interface.srv import RpcService
from booster_interface.msg import BoosterApiReqMsg
import json
import time

def create_msg(api_id, param_dict=None):
    msg = BoosterApiReqMsg()
    msg.api_id = api_id
    if param_dict is not None:
        msg.body = json.dumps(param_dict)
    else:
        msg.body = ""
    return msg


# Pitch (up/down):

# Maximum down: 1.0 rad
# Maximum up: -0.3 rad
# Range: -0.3 to 1.0 rad
# Yaw (left/right):

# Maximum left: 0.785 rad (approximately 45°)
# Maximum right: -0.785 rad (approximately -45°)
# Range: -0.785 to 0.785 rad


def main():
    rclpy.init()
    node = Node('simple_head_test_node')
    client = node.create_client(RpcService, 'booster_rpc_service')

    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')
        if not rclpy.ok():
            node.get_logger().error('Interrupted while waiting for the service. Exiting.')
            return

    # Look down
    node.get_logger().info('Looking down (1.0)...')
    req_look_down = create_msg(2004, {"pitch": 1.0, "yaw": 0.0})
    request = RpcService.Request()
    request.msg = req_look_down
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Look down result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')
    
    time.sleep(1.5)

    # Back to center
    node.get_logger().info('Centering head...')
    req_center = create_msg(2004, {"pitch": 0.0, "yaw": 0.0})
    request.msg = req_center
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Center result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')

    time.sleep(1.0)


    # Look up
    node.get_logger().info('Looking up (-0.29)...')
    req_look_up = create_msg(2004, {"pitch": -0.29, "yaw": 0.0})
    request.msg = req_look_up
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Look up result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')

    time.sleep(0.7)

    # Back to center
    node.get_logger().info('Centering head...')
    req_center = create_msg(2004, {"pitch": 0.0, "yaw": 0.0})
    request.msg = req_center
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Center result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')
    
    time.sleep(0.7)

    # Look right
    node.get_logger().info('Looking right (yaw -0.785)...')
    req_look_right = create_msg(2004, {"pitch": 0.0, "yaw": -0.785})
    request.msg = req_look_right
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Look right result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')

    time.sleep(0.7)


    # Back to center
    node.get_logger().info('Centering head...')
    req_center = create_msg(2004, {"pitch": 0.0, "yaw": 0.0})
    request.msg = req_center
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Center result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')
    
    time.sleep(0.7)



    # Look left
    node.get_logger().info('Looking left (yaw 0.785)...')
    req_look_left = create_msg(2004, {"pitch": 0.0, "yaw": 0.785})
    request.msg = req_look_left
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Look left result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')

    time.sleep(0.7)

    # Back to center
    node.get_logger().info('Centering head...')
    request.msg = req_center
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Center result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')

    node.get_logger().info('Head movement sequence completed!')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
