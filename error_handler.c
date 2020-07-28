#include <errno.h>

void check_and_set_popsize(char const *argv[], int input_id, int &NP)
{
    //Handling possible errors
    char *p; long conv;
    errno = 0;
    conv = strtol(argv[input_id], &p, 10);
    if (errno != 0 || *p != '\0' || conv < 4 || conv > 1000) 
    {
        printf("Error %d, invalid size for population %d. Input argument %d must be an integer in the interval [4,1000]\n",input_id,input_id,input_id);
        exit(EXIT_FAILURE);
    } 
    else 
        NP = (int)conv;
}

void check_and_set_Rseed(char const *argv[], int input_id, float &Rseed)
{
    //Handling possible errors
    char *p; double conv;
    errno = 0;
    conv = strtod(argv[input_id], &p);
    if (errno != 0 || *p != '\0' || conv < 0.0 || conv > 1.0) 
    {
        printf("Error %d, RNG seed out of range: input argument %d must be a float in the interval [0.0,1.0]\n",input_id,input_id);
        exit(EXIT_FAILURE);
    } 
    else 
        Rseed = (float)conv;
}

// void check_and_set_FunctionID(char const *argv[], int input_id, int &ID)
// {
//     //Handling possible errors
//     char *p; long conv;
//     errno = 0;
//     conv = strtol(argv[input_id], &p, 10);
//     if (errno != 0 || *p != '\0' || conv < 1 || conv > 15) 
//     {
//         printf("Error %d, not available objective function: input argument %d must be an interval in the interval [1,15]\n",input_id,input_id);
//         exit(EXIT_FAILURE);
//     } 
//     else 
//         ID = (int)conv;
// }
