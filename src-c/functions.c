#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include "functions.h"


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
int build_file(FILE* file, int dsize, int dec, int number_of_files, int gainValue){

    fwrite(&dsize,sizeof(dsize),1,file); //nb of samples
    fwrite(&dec,sizeof(dec),1,file);
    fwrite(&number_of_files,sizeof(number_of_files),1,file);
    fwrite(&gainValue,sizeof(gainValue),1,file);

    return 0;
}


int create_file(char* file_name, FILE *fp, int dsize, int dec, int number_of_files, int gainValue){
    if(fp){
        perror("file already exists!!");
        return -1; // file exists
    }else{
        FILE *fp = fopen(file_name, "r");
        if (fp){//if exists
            perror("file already exists");
            return -1;
        }else{
            FILE *fp = fopen(file_name, "wb+");
            build_file(fp, dsize, dec, number_of_files, gainValue);
    }
    return 1;
    }}
    
int add_to_file(FILE *file, int16_t *ram, int nb_of_Samples){
    if(file != NULL){

    fwrite(ram,sizeof(int16_t),nb_of_Samples,file);
    }else{
        perror("error open file to add samples");
        return -1;
    }
}