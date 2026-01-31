using UnityEngine;
using static GameTheory;

public class PayoffFunction 
{
    [Header("Payoff Weights")]
    public float goalWeight = 12;
    public float safetyWeight = 5f;
    public float batteryWeight = 2f;
    public float collisionPenalty = -100f;

    public float ComputePayoff(
      DroneAction droneAction,
      VisionState visionState,
      bool isExploring,   // ← ADD THIS
      Vector3 currentPos,
      Vector3 predictedPos,
      Vector3 goalPos,
      float initialDistance,
      float distanceToObstacle,
      float batteryUsed,
      float totalBattery,
      bool collision)
    {
        float payoff = 0f;

        // 1. Goal progress
        float currentDistance = Vector3.Distance(currentPos, goalPos);
        float predictedDistance = Vector3.Distance(predictedPos, goalPos);

        // Positive if we move closer, negative if we move away
        float delta = currentDistance - predictedDistance;

        // Normalize
        float normalizedDelta = delta / initialDistance;

        // Strong incentive to approach goal
        payoff += normalizedDelta * goalWeight * 2f;
        Debug.Log($"Delta: {delta}, Normalized Delta: {normalizedDelta}, Goal Payoff: {normalizedDelta * goalWeight * 2f}");

        // 2. Battery efficiency
        float batteryRemaining = 1f - (batteryUsed / totalBattery);
        payoff += batteryRemaining * batteryWeight;

        // 3. Safety based on vision
        switch (droneAction)
        {
            case DroneAction.MoveForward:
                payoff += visionState.forwardClear ? safetyWeight + 7f : -15f;
                break;

            case DroneAction.TurnLeft:
                payoff += visionState.leftClear ? safetyWeight : -5f;
                break;

            case DroneAction.TurnRight:
                payoff += visionState.rightClear ? safetyWeight : -5f;
                break;

            case DroneAction.TurnAround:
                payoff += 1f;
                break;
        }
        if (isExploring)
            payoff += 6f;   // try 5–10
        else
            payoff -= 1f;   // discourage loops

        // 4. Action cost
        payoff -= GetActionCost(droneAction);

        // 5. Collision penalty
        if (collision)
            payoff += collisionPenalty;

        // 6. Stop penalty
        if (droneAction == DroneAction.Stop)
            payoff -= 3f;

        return payoff;
    }


    float GetActionCost(DroneAction action)
    {
        switch (action)
        {
            case DroneAction.MoveForward:
                return 2f; // Moving costs battery
            case DroneAction.TurnLeft:
            case DroneAction.TurnRight:
                return 0.5f; // Turning costs less
            case DroneAction.TurnAround:
                return 1f; // 180° turn costs more
            case DroneAction.Stop:
                return 0.1f; // Stopping costs almost nothing
            default:
                return 1f;
        }
    }
}
