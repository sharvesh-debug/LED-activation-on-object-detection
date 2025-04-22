Hardware Requirements:

Raspberry Pi 3 B+
Web camera (USB webcam)
3 LEDs (different colors recommended)
3 resistors (220-330 ohm) for the LEDs
Breadboard and jumper wires
Power supply for the Raspberry Pi

Hardware Setup:

Connect the LEDs to GPIO pins:

Connect the first LED's anode (+) to GPIO 17 through a 220-330 ohm resistor(optional)
Connect the second LED's anode (+) to GPIO 27 through a 220-330 ohm resistor(optional)
Connect the third LED's anode (+) to GPIO 22 through a 220-330 ohm resistor(optional)
Connect all LED cathodes (-) to a ground (GND) pin on the Raspberry Pi

How works?
when you show one finger glows first LED, two fingers glows second LED, three fingers third LED.
LED glows for 5 second and again object detection starts
