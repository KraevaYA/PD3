#include <stdio.h>
#include <assert.h>

#include "IOdata.hpp"

void read_ts(char *file_name, float *data)
{

    FILE * file = fopen(file_name, "rt");
    assert(file != NULL);

    int i = 0;

    while (!feof(file))
    {
        fscanf(file, "%f\n", &data[i]);
        i++;
    }

    fclose(file);
}
