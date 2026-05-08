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

def main():
    rclpy.init()
    node = Node('simple_move_test_node')
    client = node.create_client(RpcService, 'booster_rpc_service')

    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')
        if not rclpy.ok():
            node.get_logger().error('Interrupted while waiting for the service. Exiting.')
            return

    # Move forward 0.5 meters (assuming velocity of 0.5 m/s for 1 second)
    node.get_logger().info('Moving forward 0.5 meters...')
    req_move_forward = create_msg(2001, {"vx": 0.5, "vy": 0.0, "vyaw": 0.0})  # kMove
    request = RpcService.Request()
    request.msg = req_move_forward
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Move forward result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')
    
    # Move for 1 second to travel approximately 0.5 meters
    time.sleep(1.0)

    # Stop after forward movement
    node.get_logger().info('Stopping after forward movement...')
    req_stop = create_msg(2001, {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
    request.msg = req_stop
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Stop result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')
    
    time.sleep(0.5)

    # Rotate 90 degrees clockwise (negative yaw velocity)
    # 90 degrees = π/2 radians ≈ 1.57 radians
    # Using angular velocity of 0.785 rad/s for 2 seconds = 1.57 radians
    node.get_logger().info('Rotating 90 degrees clockwise...')
    req_rotate = create_msg(2001, {"vx": 0.0, "vy": 0.0, "vyaw": -0.785})  # Negative for clockwise
    request.msg = req_rotate
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Rotate result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')
    
    # Rotate for 2 seconds to achieve 90 degrees
    time.sleep(2.0)

    # Stop after rotation
    node.get_logger().info('Stopping after rotation...')
    request.msg = req_stop
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Final stop result: %s' % future.result().msg.body)
    else:
        node.get_logger().error('Failed to call rpc service')

    node.get_logger().info('Movement sequence completed!')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
