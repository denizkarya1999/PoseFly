#define RED_LED    D1  // Red LED for OOK identification
#define GREEN_LED1 D2  // Green LEDs for quick-link
#define GREEN_LED2 D3  
#define GREEN_LED3 D4  

// Define OOK timing (Adjust based on camera rolling shutter settings)
#define OOK_CYCLE_TIME 250  // Time in microseconds (~4kHz frequency)
#define QUICK_LINK_CYCLE_TIME 83  // Time in microseconds (~12kHz frequency)

// Define a sample OOK pattern (Drone ID = 0b1001 = ON, OFF, OFF, ON)  
bool ook_pattern[] = {0, 1, 0, 0};   // #9
int ook_size = sizeof(ook_pattern) / sizeof(ook_pattern[0]);

void setup() {
    pinMode(RED_LED, OUTPUT);
    pinMode(GREEN_LED1, OUTPUT);
    pinMode(GREEN_LED2, OUTPUT);
    pinMode(GREEN_LED3, OUTPUT);
}

void loop() {
    // Send OOK Signal for Drone Identification (Red LED)
    for (int i = 0; i < ook_size; i++) {
        digitalWrite(RED_LED, ook_pattern[i]);
        delayMicroseconds(OOK_CYCLE_TIME);
    }
    
    // Quick-link data transmission using the Green LEDs
    for (int i = 0; i < 3; i++) {  
        digitalWrite(GREEN_LED1, HIGH);
        digitalWrite(GREEN_LED2, LOW);
        digitalWrite(GREEN_LED3, HIGH);
        delayMicroseconds(QUICK_LINK_CYCLE_TIME);
        
        digitalWrite(GREEN_LED1, LOW);
        digitalWrite(GREEN_LED2, HIGH);
        digitalWrite(GREEN_LED3, LOW);
        delayMicroseconds(QUICK_LINK_CYCLE_TIME);
    }
}

// 0001  {1, 0, 0, 1} Drone No. 5
// 0100 4  {1, 0, 1, 0} Drone No. 6
// 0011 3 {0, 0, 1, 1} Drone No. 1
// 0101 5 {0, 1, 0, 1} Drone No. 2
// 0110 6 {0, 1, 1, 0} Drone No. 3
// 0111 7 {0, 1, 1, 1} Drone No. 4

