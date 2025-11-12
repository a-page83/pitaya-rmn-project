#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>



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
int build_file(FILE* file, int dsize, int dec, int number_of_files, int gainValue);

int create_file(char* file_name, FILE *fp, int dsize, int dec, int number_of_files, int gainValue);

int add_to_file(FILE *file, int16_t *ram, int nb_of_Samples);

