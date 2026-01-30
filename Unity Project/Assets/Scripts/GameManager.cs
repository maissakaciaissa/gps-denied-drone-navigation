using UnityEngine;

public class GameManager : MonoBehaviour
{
    public DroneController drone;
    public DroneVision vision;

    [Header("Settings")]
    public bool autoNavigate = false;
    public float moveDistance = 2f;
    public float arrivalThreshold = 0.5f; // How close = "arrived"

    [Header("Timing")]
    public float pauseDuration = 1f; // Pause after reaching destination

    private Vector3 targetPosition;
    private bool isMovingToTarget = false;
    private bool isPaused = false;
    private float pauseTimer = 0f;


    

    void Start()
    {
        targetPosition = transform.position;
    }

    void Update()
    {
        // Toggle auto navigation
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

        // Manual info display
        if (Input.GetKeyDown(KeyCode.I))
        {
            DisplayVisionInfo();
        }
    }

    void NavigateWithPause()
    {
        if (isPaused)
        {
            // Currently paused - count down
            drone.Stop(); // Make sure drone stays stopped
            pauseTimer += Time.deltaTime;
            
            if (pauseTimer >= pauseDuration)
            {
                // Pause finished
                isPaused = false;
                isMovingToTarget = false;
                Debug.Log("Pause complete - deciding next move");
            }
        }
        else if (isMovingToTarget)
        {
            // Currently moving to target
            Vector3 dronePosition = drone.transform.position;   
            float distanceToTarget = Vector3.Distance(dronePosition , targetPosition);

            if (distanceToTarget <= arrivalThreshold)
            {
                // Reached destination - start pause
                drone.Stop();
                isPaused = true;
                pauseTimer = 0f;
                isMovingToTarget = false;
                Debug.Log($"Arrived at destination - pausing for {pauseDuration}s");
            }
            else
            {
                // Keep calling MoveTo every frame until we arrive
                drone.MoveTo(targetPosition);
            }
        }
        else
        {
            // Not moving, not paused - decide next move
            DecideNextMove();
        }
    }

    void DecideNextMove()
    {
        if (vision.IsForwardClear(moveDistance))
        {
            // Set new target ahead
            targetPosition = drone.transform.position + drone.transform.forward * moveDistance;
            isMovingToTarget = true;
            Debug.Log($"Decision: Moving forward to {targetPosition}");
        }
        else
        {
            // Obstacle ahead - turn in place
            drone.Stop();

            if (vision.IsRightClear(moveDistance))
            {
                drone.transform.Rotate(Vector3.up, 90f);
                Debug.Log("Decision: Turned right");
            }
            else if (vision.IsLeftClear(moveDistance))
            {
                drone.transform.Rotate(Vector3.up, -90f);
                Debug.Log("Decision: Turned left");
            }
            else
            {
                drone.transform.Rotate(Vector3.up, 180f);
                Debug.Log("Decision: Turned around");
            }

            // Start pause after turning
            isPaused = true;
            pauseTimer = 0f;
        }
    }

    void DisplayVisionInfo()
    {
        string info = $"=== Vision Status ===\n";
        info += $"Moving: {isMovingToTarget}\n";
        info += $"Paused: {isPaused}\n";

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
        info += $"Forward Clear: {vision.IsForwardClear(10f)}\n";
        info += $"Right Clear: {vision.IsRightClear(10f)}\n";
        info += $"Left Clear: {vision.IsLeftClear(10f)}\n";

        Debug.Log(info);
    }

    // Visualize in Scene view
    void OnDrawGizmos()
    {
        if (isMovingToTarget)
        {
            // Draw target position
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(targetPosition, 0.5f);
            Gizmos.DrawLine(drone.transform.position, targetPosition);

            // Draw arrival threshold
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(targetPosition, arrivalThreshold);
        }

        // Draw current state (green = moving/deciding, red = paused)
        Gizmos.color = isPaused ? Color.red : Color.green;
        Gizmos.DrawWireSphere(drone.transform.position + Vector3.up * 2, 0.3f);
    }
}
