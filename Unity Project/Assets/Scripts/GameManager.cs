using System.Collections.Generic;
using UnityEngine;
using static GameTheory;

public class GameManager : MonoBehaviour
{
    public DroneController drone;
    public DroneVision vision;

    [Header("Settings")]
    public bool autoNavigate = false;
    public float moveDistance = 2f;
    public float arrivalThreshold = 0.5f;

    [Header("Timing")]
    public float pauseDuration = 1f;

    private Vector3 targetPosition;
    private bool isMovingToTarget = false;
    private bool isPaused = false;
    private float pauseTimer = 0f;

    public Vector3 goalPosition;
    private float initialDistance;

    public bool collisionDetected = false;

    MinMax minMax;
    HashSet<Vector3Int> visitedCells = new HashSet<Vector3Int>();

    Vector3Int GetCell(Vector3 pos)
    {
        return new Vector3Int(
            Mathf.RoundToInt(pos.x),
            0,
            Mathf.RoundToInt(pos.z)
        );
    }

    void Start()
    {
        targetPosition = transform.position;
        initialDistance = Vector3.Distance(drone.transform.position, goalPosition);
        minMax = new MinMax(new PayoffFunction(),drone.transform);
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            autoNavigate = !autoNavigate;
            isPaused = false;
            isMovingToTarget = false;
            drone.Stop();
            Debug.Log($"Auto Navigate: {autoNavigate}");
        }

        if (autoNavigate)
        {
            NavigateWithPause();
        }

        if (Input.GetKeyDown(KeyCode.I))
        {
            DisplayVisionInfo();
        }
    }

    void NavigateWithPause()
    {
        if (isPaused)
        {
            drone.Stop();
            pauseTimer += Time.deltaTime;

            if (pauseTimer >= pauseDuration)
            {
                isPaused = false;
                isMovingToTarget = false;
                Debug.Log("Pause complete - deciding next move");
            }
        }
        else if (isMovingToTarget)
        {
            Vector3 dronePosition = drone.transform.position;
            float distanceToTarget = Vector3.Distance(dronePosition, targetPosition);

            if (distanceToTarget <= arrivalThreshold)
            {
                drone.Stop();
                isPaused = true;
                pauseTimer = 0f;
                isMovingToTarget = false;
                Debug.Log($"Arrived at destination - pausing for {pauseDuration}s");
            }
            else
            {
                drone.MoveTo(targetPosition);
            }
        }
        else
        {
            DecideNextMoveMinMax();
        }
    }
    
    void DecideNextMoveMinMax()
    {
        // 1. Update vision
        vision.ScanForward();

        // 2. Build vision state
        VisionState visionState = GetVisionState();

        Debug.Log($"VISION → F:{visionState.forwardClear} L:{visionState.leftClear} R:{visionState.rightClear}");

        // 3. Filter actions
        List<DroneAction> availableActions = GetAvailableActions(visionState);

        if (availableActions.Count == 0)
        {
            Debug.LogWarning("No available actions! Stopping.");
            drone.Stop();
            return;
        }

        // 4. Decide using MinMax
        DroneAction chosen = minMax.Decide(
            availableActions,
            visionState,
            drone.transform.position,
            goalPosition,
            initialDistance,
            vision.closestObstacleDistance,
            drone.totalBattery - drone.currentBattery,
            drone.totalBattery,
            collisionDetected,
            visitedCells,
            moveDistance,
            true
       
        );

        Debug.Log($"MinMax chose action: {chosen}");
        // 5. Execute
        ExecuteAction(chosen);
    }

    VisionState GetVisionState()
    {
        return new VisionState
        {
            forwardClear = vision.IsForwardClear(moveDistance),
            leftClear = vision.IsLeftClear(moveDistance),
            rightClear = vision.IsRightClear(moveDistance)
        };
    }


    // NEW METHOD: Get only valid actions
    List<DroneAction> GetAvailableActions(VisionState vision)
    {
        List<DroneAction> actions = new List<DroneAction>();

        if (vision.forwardClear)
            actions.Add(DroneAction.MoveForward);

        if (vision.leftClear)
            actions.Add(DroneAction.TurnLeft);

        if (vision.rightClear)
            actions.Add(DroneAction.TurnRight);

        actions.Add(DroneAction.TurnAround);
        actions.Add(DroneAction.Stop);

        return actions;
    }


    void ExecuteAction(DroneAction action)
    {
        Debug.Log($"=== EXECUTING: {action} ===");

        switch (action)
        {
            case DroneAction.MoveForward:
                targetPosition = drone.transform.position + drone.transform.forward * moveDistance;
                isMovingToTarget = true;
                isPaused = false;               // ✅ DO NOT PAUSE
                drone.DrainBattery(drone.batteryDrainPerMove);
                Debug.Log($"Moving forward to {targetPosition}");
                visitedCells.Add(GetCell(drone.transform.position));
                return; // ⬅️ VERY IMPORTANT

            case DroneAction.TurnLeft:
                drone.TurnLeft();
                break;

            case DroneAction.TurnRight:
                drone.TurnRight();
                break;

            case DroneAction.TurnAround:
                drone.TurnAround();
                break;

            case DroneAction.Stop:
                drone.Stop();
                break;
        }

        // Only pause for NON-movement actions
        isMovingToTarget = false;
        isPaused = true;
        pauseTimer = 0f;
    }


    void DisplayVisionInfo()
    {
        string info = $"=== Vision Status ===\n";
        info += $"Moving: {isMovingToTarget}\n";
        info += $"Paused: {isPaused}\n";
        info += $"Battery: {drone.GetBatteryPercentage():F1}%\n";

        if (isMovingToTarget)
        {
            float dist = Vector3.Distance(drone.transform.position, targetPosition);
            info += $"Distance to Target: {dist:F2}m\n";
        }

        if (isPaused)
        {
            info += $"Pause Time Left: {(pauseDuration - pauseTimer):F1}s\n";
        }

        info += $"Path Clear: {vision.isPathClear}\n";
        info += $"Closest Obstacle: {vision.closestObstacleDistance:F2}m\n";
        info += $"Forward Clear: {vision.IsForwardClear(moveDistance)}\n";
        info += $"Right Clear: {vision.IsRightClear(moveDistance)}\n";
        info += $"Left Clear: {vision.IsLeftClear(moveDistance)}\n";
        info += $"Distance to Goal: {Vector3.Distance(drone.transform.position, goalPosition):F2}m\n";

        Debug.Log(info);
    }

    void OnDrawGizmos()
    {
        if (isMovingToTarget)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(targetPosition, 0.5f);
            Gizmos.DrawLine(drone.transform.position, targetPosition);
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(targetPosition, arrivalThreshold);
        }

        // Draw goal
        Gizmos.color = Color.green;
        Gizmos.DrawWireSphere(goalPosition, 1f);
        Gizmos.DrawLine(drone.transform.position, goalPosition);

        // State indicator
        Gizmos.color = isPaused ? Color.red : Color.green;
        Gizmos.DrawWireSphere(drone.transform.position + Vector3.up * 2, 0.3f);
    }
}
