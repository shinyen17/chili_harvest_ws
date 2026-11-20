# chili_harvest_ws

Chili ripeness detection + stem cutting on the cocoabot arm.
Overlay on top of cocoabot_ws (same as tomato_harvest_ws). Full guide: see the
"Chili Harvesting Robot - Setup & Code Guide" doc.

## Build (on the robot PC)
    cd ~/workspaces/chili_harvest_ws
    source /opt/ros/humble/setup.bash
    source ~/workspaces/cocoabot_ws/install/setup.bash
    cp <your weights> src/chili_harvest/models/    # chili_ripeness.pt, chili_stem_seg.pt
    colcon build --symlink-install --packages-select chili_harvest
    source install/setup.bash

## Run (4 terminals, each sourced as above + install/setup.bash)
    1) ros2 launch orbbec_camera gemini.launch.xml          # camera (name may differ)
    2) ros2 launch chili_harvest chili_robot.launch.py      # MoveIt + servo bridge
    3) ros2 launch chili_harvest chili_harvest.launch.py    # dry run by default
    4) ros2 service call /chili/enable std_srvs/srv/SetBool "{data: true}"

Live cutting: ros2 launch chili_harvest chili_harvest.launch.py dry_run:=false

## Topics
    /chili/harvest_target  PointStamped (base_link) - nearest ripe chili cut point
    /chili/detections      String JSON - every detection, label, box, cut point
    /chili/stats           String JSON - counts + inference ms
    /chili/debug_image     Image - annotated frame (rqt_image_view)
    /chili/harvest_status  String - controller state
Cycle log: ~/chili_harvest_logs/cycles.csv

## Unit tests (no ROS needed)
    cd src/chili_harvest && python3 -m pytest test
