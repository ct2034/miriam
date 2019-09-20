sqrt2_2=$(python -c "import math;print math.sqrt(2)/2")
rostopic pub -1 /robot_0/move_base_simple/goal geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: 'map'
pose:
  position:
    x: 1.0
    y: 3.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: -$sqrt2_2
    w: $sqrt2_2" &
rostopic pub -1 /robot_4/move_base_simple/goal geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: 'map'
pose:
  position:
    x: -1.0
    y: -3.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: $sqrt2_2
    w: $sqrt2_2"
