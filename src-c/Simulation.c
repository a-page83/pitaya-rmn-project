#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "rp.h"

int main (int argc, char **argv) {
    int unsigned led = RP_LED0;

    float excitation_duration_microseconds = 41.027;
    float excitation_amplitude_Volts = 0.05;
    float answer_amplitude_Volts = 0.5;
    float Larmor_frequency_Hertz = 24378040.422;
    int duration_burst_second = 1;
    float precession_frequency_shifted = 1000;
    int number_burst_cycle = Larmor_frequency_Hertz*duration_burst_second;
    
    printf("number burst %d \n",number_burst_cycle);

    if (argc > 1) {
        led = atoi(argv[1]);
    }

    printf("Blinking LED[0]\n");
    printf("Emulation started\n");

    
    // Initialization of API
    if (rp_Init() != RP_OK) {
        fprintf(stderr, "Red Pitaya API init failed!\n");
        return EXIT_FAILURE;
   }

 	rp_GenReset();
	rp_GenWaveform(RP_CH_1, RP_WAVEFORM_DC);
	rp_GenAmp(RP_CH_1, 0.5);
	


    while(1){

        //Attente de la salve d'excitation
        //led indiquant que la simulation tourne
        rp_DpinSetState(led+1, RP_HIGH);

        rp_GenOutEnable(RP_CH_1);
        rp_GenTriggerOnly(RP_CH_1); //déclenchement out1 NOW     
        
        rp_DpinSetState(led, RP_LOW);
        rp_DpinSetState(led+1, RP_LOW);
        
        usleep(3000);
        rp_GenOutDisable(RP_CH_1);
        usleep(100000);
    }
    // Releasing resources
    rp_Release();

    return EXIT_SUCCESS;
}
