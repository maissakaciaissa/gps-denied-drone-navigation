using UnityEngine;

public class DroneController : MonoBehaviour
{
    [Header("Movement Settings")]
    public float moveSpeed = 5f;
    public float rotationSpeed = 100f;
    public float hoverForce = 9.81f;

    private Rigidbody rb;
    private bool isAutoMoving = false; // Track if MoveTo is controlling
    private Vector3 autoMoveTarget;

    [Header("Battery Settings")]
    public float totalBattery = 1000f;
    public float currentBattery = 100f;
    public float batteryDrainPerMove = 2f;      // Battery cost to move forward
    public float batteryDrainPerTurn = 0.5f;    // Battery cost to turn
    



    private void Awake()
    {
        rb = GetComponent<Rigidbody>();

        // High damping for instant stop
        rb.linearDamping = 10f;
        rb.angularDamping = 10f;

        currentBattery = totalBattery;
    }

    private void FixedUpdate()
    {
        // Hover
        rb.AddForce(Vector3.up * hoverForce, ForceMode.Acceleration);

        if (isAutoMoving)
        {
            // Auto movement via MoveTo
            AutoMove();
        }
        else
        {
            // Manual control only when not auto-moving
            ManualControl();
        }

        // Check battery level
        if (currentBattery <= 0)
        {
            Debug.LogWarning("BATTERY DEPLETED!");
            Stop();
        }
    }
  

    public void TurnLeft(float angle = 90f)
    {
        if (!HasBattery(batteryDrainPerTurn))
        {
            Debug.LogWarning("Not enough battery to turn!");
            return;
        }

        transform.Rotate(Vector3.up, -angle);
        DrainBattery(batteryDrainPerTurn);
    }

    public void TurnRight(float angle = 90f)
    {
        if (!HasBattery(batteryDrainPerTurn))
        {
            Debug.LogWarning("Not enough battery to turn!");
            return;
        }

        transform.Rotate(Vector3.up, angle);
        DrainBattery(batteryDrainPerTurn);
    }

    public void TurnAround()
    {
        if (!HasBattery(batteryDrainPerTurn * 2))
        {
            Debug.LogWarning("Not enough battery to turn around!");
            return;
        }

        transform.Rotate(Vector3.up, 180f);
        DrainBattery(batteryDrainPerTurn * 2); // 180° turn costs more
    }
    void ManualControl()
    {
        float h = Input.GetAxis("Horizontal");
        float v = Input.GetAxis("Vertical");

        if (Mathf.Abs(h) > 0.01f || Mathf.Abs(v) > 0.01f)
        {
            // Move
            Vector3 movement = (transform.forward * v + transform.right * h).normalized;
            rb.linearVelocity = new Vector3(
                movement.x * moveSpeed,
                rb.linearVelocity.y,
                movement.z * moveSpeed
            );

            DrainBattery(batteryDrainPerMove);
        }
        else
        {
            // Stop horizontal movement immediately
            rb.linearVelocity = new Vector3(0, rb.linearVelocity.y, 0);
        }

        // Rotation
        if (Input.GetKey(KeyCode.Q))
        {
            transform.Rotate(Vector3.up, -rotationSpeed * Time.fixedDeltaTime);
        }
        else if (Input.GetKey(KeyCode.E))
        {
            transform.Rotate(Vector3.up, rotationSpeed * Time.fixedDeltaTime);
        }

        // Stop rotation when not rotating
        if (!Input.GetKey(KeyCode.Q) && !Input.GetKey(KeyCode.E))
        {
            rb.angularVelocity = Vector3.zero;
        }
    }

    void AutoMove()
    {
        // Move toward target
        Vector3 direction = (autoMoveTarget - transform.position).normalized;
        rb.linearVelocity = new Vector3(
            direction.x * moveSpeed,
            rb.linearVelocity.y,
            direction.z * moveSpeed
        );
        DrainBattery(batteryDrainPerMove * Time.fixedDeltaTime);
    }

    public void MoveTo(Vector3 targetPosition)
    {
        autoMoveTarget = targetPosition;
        isAutoMoving = true;
    }

    public void Stop()
    {
        isAutoMoving = false;
        rb.linearVelocity = new Vector3(0, rb.linearVelocity.y, 0);
        rb.angularVelocity = Vector3.zero;
    }

    public bool IsMoving()
    {
        Vector3 horizontalVelocity = new Vector3(rb.linearVelocity.x, 0, rb.linearVelocity.z);
        return horizontalVelocity.magnitude > 0.1f;
    }

     public void DrainBattery(float amount)
    {
        currentBattery = Mathf.Max(0, currentBattery - amount);
    }
    
    public bool HasBattery(float amount)
    {
        return currentBattery >= amount;
    }
    
    public float GetBatteryPercentage()
    {
        return (currentBattery / totalBattery) * 100f;
    }
    
}
