using System.Collections.Generic;
using UnityEngine;

public class DroneVision : MonoBehaviour
{
    [Header("Vision Settings")]
    public float visionRange = 10f;
    public int horizontalRays = 5;  // Number of rays horizontally
    public int verticalRays = 3;    // Number of rays vertically
    public float horizontalFOV = 60f; // Field of view in degrees (horizontal)
    public float verticalFOV = 30f;   // Field of view in degrees (vertical)
    public LayerMask obstacleLayer;
    public float visionDistance = 3f;
    [Header("Debug")]
    public bool showVisionRays = true;
    public Color clearColor = Color.green;
    public Color blockedColor = Color.red;



    [Header("Vision Data")]
    public bool isPathClear = true;
    public float closestObstacleDistance = float.MaxValue;
    public List<RaycastHit> detectedObstacles = new List<RaycastHit>();

    void Update()
    {
        ScanForward();
    }

    public void ScanForward()
    {
        detectedObstacles.Clear();
        isPathClear = true;
        closestObstacleDistance = visionRange;

        // Starting angles
        float startHorizontalAngle = -horizontalFOV / 2f;
        float startVerticalAngle = -verticalFOV / 2f;

        float horizontalStep = horizontalRays > 1 ? horizontalFOV / (horizontalRays - 1) : 0;
        float verticalStep = verticalRays > 1 ? verticalFOV / (verticalRays - 1) : 0;

        // Cast rays in a grid pattern
        for (int v = 0; v < verticalRays; v++)
        {
            for (int h = 0; h < horizontalRays; h++)
            {
                float horizontalAngle = startHorizontalAngle + (h * horizontalStep);
                float verticalAngle = startVerticalAngle + (v * verticalStep);

                // Calculate ray direction
                Vector3 direction = CalculateRayDirection(horizontalAngle, verticalAngle);

                // Cast ray
                RaycastHit hit;
                if (Physics.Raycast(transform.position, direction, out hit, visionRange, obstacleLayer))
                {
                    // Obstacle detected
                    isPathClear = false;
                    detectedObstacles.Add(hit);

                    if (hit.distance < closestObstacleDistance)
                    {
                        closestObstacleDistance = hit.distance;
                    }

                    if (showVisionRays)
                        Debug.DrawRay(transform.position, direction * hit.distance, blockedColor);
                }
                else
                {
                    // No obstacle
                    if (showVisionRays)
                        Debug.DrawRay(transform.position, direction * visionRange, clearColor);
                }
            }
        }
    }


    bool CastRay(Vector3 direction, float distance)
    {
        RaycastHit hit;
        if (Physics.Raycast(transform.position, direction, out hit, distance, obstacleLayer))
        {
            closestObstacleDistance = hit.distance;
            isPathClear = false;
            return false;
        }

        closestObstacleDistance = distance;
        isPathClear = true;
        return true;
    }

    Vector3 CalculateRayDirection(float horizontalAngle, float verticalAngle)
    {
        // Rotate around Y axis for horizontal angle
        Quaternion horizontalRotation = Quaternion.AngleAxis(horizontalAngle, transform.up);

        // Rotate around right axis for vertical angle
        Quaternion verticalRotation = Quaternion.AngleAxis(verticalAngle, transform.right);

        // Combine rotations and apply to forward direction
        Vector3 direction = horizontalRotation * verticalRotation * transform.forward;

        return direction.normalized;
    }
    // Check if a specific direction is clear
    public bool IsDirectionClear(Vector3 direction, float distance = -1)
    {
        if (distance < 0) distance = visionRange;

        RaycastHit hit;
        return !Physics.Raycast(transform.position, direction, out hit, distance, obstacleLayer);
    }
    public bool IsForwardClear(float distance)
    {
        return CastRay(transform.forward, distance);
    }

    public bool IsRightClear(float distance)
    {
        return CastRay(transform.right, distance);
    }

    public bool IsLeftClear(float distance)
    {
        return CastRay(-transform.right, distance);
    }


    // Get distance to obstacle in front
    public float GetForwardDistance()
    {
        RaycastHit hit;
        if (Physics.Raycast(transform.position, transform.forward, out hit, visionRange, obstacleLayer))
        {
            return hit.distance;
        }
        return visionRange;
    }

    // Simple obstacle avoidance decision
    public Vector3 GetSafeDirection()
    {
        if (IsForwardClear(5f))
        {
            return transform.forward;
        }
        else if (IsRightClear(10f))
        {
            return transform.right;
        }
        else if (IsLeftClear(10f))
        {
            return -transform.right;
        }
        else
        {
            return -transform.forward; // Go back
        }
    }

    // Visualize vision cone in Scene view
    void OnDrawGizmos()
    {
        if (!showVisionRays) return;

        

        Gizmos.color = Color.green;
        Gizmos.DrawRay(transform.position, transform.forward * visionDistance);

        Gizmos.color = Color.blue;
        Gizmos.DrawRay(transform.position, transform.right * visionDistance);

        Gizmos.color = Color.red;
        Gizmos.DrawRay(transform.position, -transform.right * visionDistance);
    }

}
