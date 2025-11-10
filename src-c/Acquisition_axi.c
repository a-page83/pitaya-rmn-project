/////// VERSION 2 - rapide dans un seul fichier ////// 
////// 10 en 60ms  

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "rp.h"

#define DATA_SIZE_MAX 524288
#define SAMPLE_RATE 125000000

/**
 * @brief Crée un fichier avec un en-tête contenant des informations d'acquisition.
 *
 * Le fichier contiendra les informations suivantes sur une première ligne :
 *  dsize, decimation, nombre de fichiers, gain, offset, nombre de bits.
 *
 * @param fichier           Pointeur vers le fichier ouvert en écriture.
 * @param dsize             Taille des données à enregistrer.
 * @param dec               Facteur de décimation utilisé lors de l'acquisition.
 * @param number_of_files   Nombre de fichiers à traiter/enregistrer.
 *
 * @return int              0 si succès, -1 si erreur lors de la récupération du gain.
 *
 * @note Le offset est actuellement fixé à 0 car la fonction rp_AcqAxiGetOffset n'est pas supportée.
 * @note Le gain est récupéré via rp_AcqGetGainV sur le canal 2 (RP_CH_2).
 */
int create_file(FILE* fichier, int dsize, int dec, int number_of_files){   

    float gainValue; //1 ou 20V
    
    float offset = 0;
    int nb_bits = 14; 
    rp_channel_t channel = RP_CH_1;

    // Fonction ne marche pas rp_AcqAxigetOffset non comprise par la RedPitaya lors de la compilation
    // if (rp_AcqAxiGetOffset(channel, &offset) != RP_OK){
    //     return -1;
    // }

    if (rp_AcqGetGainV(channel, &gainValue) != RP_OK){
        return -1;
    }

    fprintf(fichier, "%d, ", dsize );
    fprintf(fichier, "%d, ", dec );
    fprintf(fichier, "%d, ", number_of_files);
    fprintf(fichier, "%f, ",(float) gainValue);
    fprintf(fichier, "%f, ", offset);
    fprintf(fichier, "%d\n", nb_bits);

    return 0;
}

int main(int argc, char **argv)
{
    uint32_t dsize = DATA_SIZE_MAX;
    uint32_t dec; 
    float excitation_duration_seconds; //41.027e-06
    float excitation_duration_microseconds;
    float Larmor_frequency_Hertz;
    int delayRepeat_micro; //en usecondes
    int number_of_files;
    char nomFichier[256];

    float excitation_amplitude_Volts = 0.19;
    float oscillator_amplitude_Volts = 0.8;

    // Pin Acq Settings
    rp_pinState_t Gain = RP_LOW;
    rp_channel_t CH_ACQ = RP_CH_1;
    rp_acq_trig_state_t state;
    uint32_t g_adc_axi_start,g_adc_axi_size;

    /////// ----- ARGUMENTS DU PROGRAMME ----- //////
    if (argc < 7) {
    fprintf(stderr, "Erreur : nombre d'arguments insuffisant.\n");
    fprintf(stderr, "Usage : %s <dsize> <dec> <number_of_files> <nomFichier> <Frequency> <exitation duration>\n", argv[0]);
    exit(EXIT_FAILURE);
    }
    dsize = atoi(argv[1]);
    dec = atoi(argv[2]);
    number_of_files = atoi(argv[3]);
    strcpy(nomFichier, argv[4]);
    Larmor_frequency_Hertz = atof(argv[5]);
    excitation_duration_seconds = atof(argv[6]);
    delayRepeat_micro = atoi(argv[7]);
    printf("larmor %f, duration excitation %f\n",Larmor_frequency_Hertz, excitation_duration_seconds);
    // Vérification des valeurs numériques
    if (dsize <= 0 || dec < 0 || number_of_files <= 0) {
        fprintf(stderr, "Erreur : paramètres invalides.\n");
        fprintf(stderr, "dsize = %d (doit être > 0)\n", dsize);
        fprintf(stderr, "dec = %d (doit être >= 0)\n", dec);
        fprintf(stderr, "number_of_files = %d (doit être > 0)\n", number_of_files);
        exit(EXIT_FAILURE);
    }


    int excitation_burst_cycles_tot = Larmor_frequency_Hertz *excitation_duration_seconds;
    float oscillator_frequency = Larmor_frequency_Hertz + 6000;

    float *buff1 = (float *)malloc(dsize * sizeof(float));
    uint32_t posChA;
    bool fillState = false;
    

    if (rp_InitReset(false) != RP_OK) {
            fprintf(stderr, "Rp api init failed!\n");
            return -1;
    }


    //// INITIALISATION AQUISITION ///////
    if(rp_AcqReset()!=RP_OK){
        fprintf(stderr, "rp_AcqReset failed!\n");
        return -1;
    }
    if (rp_AcqAxiSetDecimationFactor(dec) != RP_OK) {
        fprintf(stderr, "rp_AcqAxiSetDecimationFactor failed!\n");
        return -1;
    }
    if (rp_AcqAxiSetTriggerDelay(CH_ACQ, dsize)  != RP_OK) { 
        fprintf(stderr, "rp_AcqAxiSetTriggerDelay CH_ACQ failed!\n");
        return -1;
    }
    if (rp_AcqSetGain(CH_ACQ, Gain) != RP_OK){
        fprintf(stderr, "rp_AcqSetGain CH1 Failed\n");
        return -1;
    }


    //// ---- INITIALISATION GENERATION BURST ---- ////
    if(rp_GenReset() != RP_OK){
            fprintf(stderr, "rp_GenReset failed!\n");
            return -1;
    } 
    if(rp_GenWaveform(RP_CH_1, RP_WAVEFORM_SINE) != RP_OK){
        fprintf(stderr, "rp_GenWaveform RP_CH_1 SINE failed!\n");
        return -1;
    }
    if(rp_GenFreq(RP_CH_1, Larmor_frequency_Hertz) != RP_OK){
        fprintf(stderr, "rp_GenFreq RP_CH_1 failed!\n");
        return -1;
    }
    if(rp_GenAmp(RP_CH_1, excitation_amplitude_Volts) != RP_OK){
        fprintf(stderr, "rp_GenAmp RP_CH_1 failed!\n");
        return -1;
    }
    if(rp_GenMode(RP_CH_1, RP_GEN_MODE_BURST) != RP_OK){
        fprintf(stderr, "rp_GenMode RP_CH_1 BURST failed!\n");
        return -1;
    }    
    if(rp_GenBurstCount(RP_CH_1, excitation_burst_cycles_tot) != RP_OK){
        fprintf(stderr, "rp_GenBurstCount RP_CH_1 failed!\n");
        return -1;
    }//valeur max pour GenBurstCount = 50 000
    if(rp_GenBurstRepetitions(RP_CH_1, 1) != RP_OK){
        fprintf(stderr, "rp_GenBurstRepetitions RP_CH_1 failed!\n");
        return -1;
    }//Répété 1 fois pour que le burst dure qq usecondes   
    if(rp_GenBurstPeriod(RP_CH_1, 1) != RP_OK){
        fprintf(stderr, "rp_GenBurstPeriod RP_CH_1 failed!\n");
        return -1;
    }//une micro seconde entre chaque répétition
    if( rp_GenOutEnable(RP_CH_1) != RP_OK){
        fprintf(stderr, "rp_GenOutEnable RP_CH_1 failed!\n");
        return -1;
    }


    //// ---- INITIALISATION ET LANCEMENT DE L'OSCILLATEUR LOCAL ---- ////
    if(rp_GenWaveform(RP_CH_2, RP_WAVEFORM_SINE) != RP_OK){
        fprintf(stderr, "rp_GenWaveform RP_CH_1 SINE failed!\n");
        return -1;
    }
    if(rp_GenFreq(RP_CH_2, oscillator_frequency) != RP_OK){
        fprintf(stderr, "rp_GenFreq RP_CH_1 failed!\n");
        return -1;
    }
    if(rp_GenAmp(RP_CH_2, oscillator_amplitude_Volts) != RP_OK){
        fprintf(stderr, "rp_GenAmp RP_CH_1 failed!\n");
        return -1;
    }
    if(rp_GenMode(RP_CH_2, RP_GEN_MODE_CONTINUOUS) != RP_OK){
        fprintf(stderr, "rp_GenMode RP_CH_1 BURST failed!\n");
        return -1;
    }
    if( rp_GenOutEnable(RP_CH_2) != RP_OK){
        fprintf(stderr, "rp_GenOutEnable RP_CH_1 failed!\n");
        return -1;
    }
  

    //// ---- CREATION DU FICHIER ---- ////
    FILE *fichier = fopen(nomFichier, "w");
    printf("fichier crée : ");
    puts(nomFichier);
    if (fichier == NULL) {
        perror("Erreur lors de l'ouverture du fichier\n");
        return -1; // Quitte le programme avec erreur
    }
    if (create_file(fichier, dsize, dec, number_of_files)){
        perror("Erreur de creation fichier");
        return -1;
    }


    //// ---- BOUCLE DE FICHIERS ---- ////
    rp_DpinSetState(RP_LED0+1, RP_HIGH);
    clock_t begin = clock();
    int i=0;
    int num_led = 1;
    for (i=0;i<number_of_files;i++){
        num_led = (i*number_of_files)/5;
        if(i%2){
            rp_DpinSetState(RP_LED0+1, RP_HIGH);}
        else{
            rp_DpinSetState(RP_LED0+1, RP_LOW);
        }
        fillState = false;
        
        rp_AcqResetFpga();
        rp_AcqAxiGetMemoryRegion(&g_adc_axi_start,&g_adc_axi_size);
        //printf("Reserved memory start 0x%X size 0x%X bytes\n",g_adc_axi_start,g_adc_axi_size);

        if (rp_AcqAxiSetBufferSamples(CH_ACQ,g_adc_axi_start, dsize) != RP_OK) {
        fprintf(stderr, "rp_AcqAxiSetBuffer CH_ACQ failed!\n");
        return -1;
        }
        if (rp_AcqAxiEnable(CH_ACQ, true)) {
            fprintf(stderr, "rp_AcqAxiEnable CH_ACQ failed!\n");
            return -1;
        }

        ////////////DECLENCHEMENT SYNCHRONISE///////////////
        if(rp_GenSynchronise() != RP_OK){
            fprintf(stderr, "rp_GenSynchronise failed!\n");
            return -1;
        }
        usleep(excitation_duration_microseconds);

        //LANCEMENT, DECLENCGEMENT DE L'AQUISITION AVANT LE BURST
        if (rp_AcqStart() != RP_OK) {
        fprintf(stderr, "rp_AcqStart failed!\n");
        return -1;
        }
        if( rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW) != RP_OK){ //Possible de mettre RP_TRIG_SRC_CHD_PE pour déclencher aqu sur le front montant de l'excitation
            fprintf(stderr, "rp_AcqSetTriggerSrc RP_TRIG_SRC_NOW failed!\n");
            return -1;
        }

        // ATTENTE DU DECLENCHEMENT DU TRIGGER
        while(1){
            rp_AcqGetTriggerState(&state);
            if(state == RP_TRIG_STATE_TRIGGERED){
                usleep(1);
                break;
            }
        }
        


        printf ("wait to be filled\n");
        while (!fillState) {
            if (rp_AcqAxiGetBufferFillState(CH_ACQ, &fillState) != RP_OK) {
                fprintf(stderr, "rp_AcqAxiGetBufferFillState CH_ACQ failed!\n");
                return -1;
            }
        }
        
        if(rp_AcqStop() != RP_OK){
                fprintf(stderr, "rp_AcqStop failed!\n");
                return -1;
        }

        if(rp_AcqAxiGetWritePointerAtTrig(CH_ACQ, &posChA)!=RP_OK){
            fprintf(stderr,"rp_AcqAxiGetWritePointerAtTrig Error");
        }
        printf("Tr pos1: 0x%x\n",posChA);

        if(rp_AcqAxiGetDataV(CH_ACQ, posChA, &dsize, buff1)!=RP_OK){
            fprintf(stderr, "rp_AcqAxiGetDataV failed\n");
        }
        

        //////  Ecriture des données dans le fichier    //////
        printf("ecriture FID %d\n",i);
        for (int i = 0; i < dsize; i++) {
            //printf("[%d]\t%f\n",i,buff1[i]);
            fprintf(fichier, "%f", buff1[i]);
            if (i!= dsize -1) fprintf(fichier, ",");
        }
        fprintf(fichier, "\n");

        usleep(delayRepeat_micro);

    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    rp_DpinSetState(RP_LED0+1, RP_LOW);

    printf("Temps d'execution : %lf\n",time_spent);

    fclose(fichier);

    /* Releasing resources */
    rp_AcqAxiEnable(CH_ACQ, false);
    rp_Release();
    free(buff1);
    return 0;
}
