using UnityEngine;

public class GameTheory {
    public struct VisionState
    {
        public bool forwardClear;
        public bool leftClear;
        public bool rightClear;
    }


    // Drone actions (what the drone can do)
    public enum DroneAction
    {
        MoveForward,
        TurnLeft,
        TurnRight,
        TurnAround,
        Stop
    }
}
