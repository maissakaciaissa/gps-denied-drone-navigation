using System.Collections.Generic;
using UnityEngine;
using static GameTheory;

public class MinMax
{
    PayoffFunction payoff;
    private Transform droneTransform; // Need drone's transform for rotation

    public MinMax(PayoffFunction p, Transform drone)
    {
        payoff = p;
        droneTransform = drone;
    }

    public DroneAction Decide(
        List<DroneAction> actions,
        VisionState vision,
        Vector3 currentPos,
        Vector3 goalPos,
        float initialDistance,
        float distanceToObstacle,
        float batteryUsed,
        float totalBattery,
        bool collision,
        HashSet<Vector3Int> visited,
        float moveDistance, // Add this parameter
        bool debug = false)
    {
        DroneAction bestAction = DroneAction.Stop;
        float bestScore = float.NegativeInfinity;

        if (debug)
        {
            Debug.Log("========== MINIMAX DECISION ==========");
            Debug.Log($"Current Position: {currentPos}");
            Debug.Log($"Current Cell: {ToCell(currentPos)}");
            Debug.Log($"Visited Cells: {visited.Count}");
            Debug.Log($"Drone Facing: {droneTransform.forward}");
        }

        foreach (var action in actions)
        {
            Vector3 predictedPos = PredictPosition(action, currentPos, moveDistance);
            Vector3Int predictedCell = ToCell(predictedPos);
            bool exploring = !visited.Contains(predictedCell);

            float score = payoff.ComputePayoff(
                action,
                vision,
                exploring,
                currentPos,
                predictedPos,
                goalPos,
                initialDistance,
                distanceToObstacle,
                batteryUsed,
                totalBattery,
                collision
            );

            if (debug)
            {
                string exploringMark = exploring ? "🆕" : "🔄";
                Debug.Log($"  {action,-15} → {predictedCell,15} {exploringMark} score={score,7:F2}");
            }

            if (score > bestScore)
            {
                bestScore = score;
                bestAction = action;
            }
        }

        if (debug)
        {
            Debug.Log($">>> CHOSEN: {bestAction} (score={bestScore:F2})");
            Debug.Log("=====================================\n");
        }

        return bestAction;
    }

    Vector3 PredictPosition(DroneAction action, Vector3 currentPos, float moveDistance)
    {
        Vector3 predictedPos = currentPos;

        Debug.Log("currentPos: " + currentPos);

        switch (action)
        {
            case DroneAction.MoveForward:
                // Use drone's ACTUAL forward direction (not world forward)
                predictedPos += droneTransform.forward * moveDistance;
                Debug.Log($"Predicted Position after MoveForward: {predictedPos}");
                break;

            case DroneAction.TurnLeft:
                predictedPos += -1 * droneTransform.right * moveDistance;
                Debug.Log($"Predicted Position after TurnLeft: {predictedPos}");
                break;
            case DroneAction.TurnRight:
                predictedPos += droneTransform.right * moveDistance;
                Debug.Log($"Predicted Position after TurnRight: {predictedPos}");
                break;
            case DroneAction.TurnAround:
                // Turning doesn't change position
                predictedPos = currentPos;
                break;

            case DroneAction.Stop:
                // No movement
                predictedPos = currentPos;
                break;
        }

        return predictedPos;
    }

    Vector3Int ToCell(Vector3 pos)
    {
        // Use consistent cell size (should match your grid)
        float cellSize = 1f;
        return new Vector3Int(
            Mathf.RoundToInt(pos.x / cellSize),
            0,
            Mathf.RoundToInt(pos.z / cellSize)
        );
    }
}