/*=========================================================================
 *=========================================================================
  == [DISCLAIMER]: THIS SOFTWARE AND ANY ACCOMPANYING DOCUMENTATION IS   ==
  == RELEASED "AS IS".  THE U.S. GOVERNMENT MAKES NO WARRANTY OF ANY     ==
  == KIND, EXPRESS OR IMPLIED, CONCERNING THIS SOFTWARE AND ANY          ==
  == ACCOMPANYING DOCUMENTATION, INCLUDING, WITHOUT LIMITATION, ANY      ==
  == WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  ==
  == IN NO EVENT WILL THE U.S. GOVERNMENT BE LIABLE FOR ANY DAMAGES      ==
  == ARISING OUT OF THE USE, OR INABILITY TO USE, THIS SOFTWARE OR ANY   ==
  == ACCOMPANYING DOCUMENTATION, EVEN IF INFORMED IN ADVANCE OF THE      ==
  == POSSIBLITY OF SUCH DAMAGES.                                         ==
  =========================================================================
  =========================================================================*/

/*-------------------------------------------------------------------------
 *                        Routine: mstar2raw (Version 2.0)
 *                         Author: John F. Querns, Veridian Engineering
 *                           Date: 25 September 1998
 *
 * What's new:
 *
 *  (1) Added code to check current CPU for byte ordering. If little-endian,
 *      code automatically byteswaps input data before any further processing
 *      is done.
 *
 *  (2) Added Magnitude data output only option.
 *
 *-------------------------------------------------------------------------
 *
 * Purpose: This routine inputs MSTAR fullscene and target chip images
 *          and outputs:
 *
 *          MSTAR TARGET CHIPS:
 *
 *             32-bit float Magnitude + 32-bit float phase
 *             32-bit float Magnitude
 *
 *          MSTAR FULLSCENES (including Clutter scenes):
 *
 *             16-bit UINT Magnitude + 16-bit UINT (12-bits signif) Phase
 *             16-bit UINT Magnitude
 *
 *          MSTAR Phoenix (Summary) header as an ASCII file.
 *
 *-------------------------------------------------------------------------
 *
 * [Calls]:
 *
 *     float
 *     byteswap_SR_IR()      -- Does big-endian to little-endian float
 *                              byteswap..this is specifically for the
 *                              case of Sun big-endian to PC-Intel
 *                              little-endian data.
 *
 *     unsigned short
 *     byteswap_SUS_IUS()    -- Does big-endian to little-endian swap for
 *                              unsigned short (16-bit) numbers. This is
 *                              specifically for the case of Sun big-
 *                              endian to PC-Intel little-endian data.
 *
 *     int
 *     CheckByteOrder()      -- This checks the byte order for the CPU that
 *                              this routine is compiled run on. If the
 *                              CPU is little-endian, it will return a 0
 *                              value (LSB_FIRST); else, it will return a 1
 *                              (MSB_FIRST).
 *
 *                              Taken from:
 *
 *                                Encyclopedia of Graphic File
 *                                Formats, Murray & Van Ryper,
 *                                O'Reilly & Associates, 1994,
 *                                pp. 114-115.
 *
 *------------------------------------------------------------------------
 *
 * [Syntax/Usage]:
 *
 *       mstar2raw <MSTAR Input> [Output Option] [enter]
 *
 *         where:
 *               Output Option = [0] --> Output all data (MAG+PHASE)
 *                               [1] --> Output MAG data only
 *
 *
 *      Example 1: Generate RAW binary image and ASCII headers for MSTAR
 *                 fullscene image: hb00001. Output ALL data for output
 *                 image file.
 *
 *                 % mstar2raw hb00001 0 [enter]
 *
 *                 Example 1 will generate the following output files:
 *
 *                 hb00001.all   <-- RAW binary 16-bit mag+12-bit phase data
 *                 hb00001.hdr   <-- ASCII Phoenix (Summary) header
 *
 *     Example 2:  Generate RAW binary image and ASCII headers for MSTAR
 *                 target chip image: hb3900.0015.  Output ALL data for
 *                 output image file.
 *
 *                 % mstar2raw hb3900.0015 0 [enter]
 *
 *                 Example 2 will generate the following output files:
 *
 *                 hb3900.0015.all <-- RAW 32-bit float mag+phase data
 *                 hb3900.0015.hdr <-- ASCII Phoenix (Summary) header
 *
 *                 NOTE: The MSTAR target chip data is float data be-
 *                       cause it is calibrated.  See the file,
 *                       "MSTAR.txt", for an explanation.
 *
 *     Example 3: Generate RAW binary image and ASCII headers for MSTAR
 *                fullscene clutter image "hb12345". Output ONLY the
 *                magnitude data in the RAW output image.
 *
 *                 % mstar2raw hb12345 1 [enter]
 *
 *                 Example 3 will generate the following output files:
 *
 *                 hb12345.mag   <-- RAW binary 16-bit magnitude data
 *                 hb12345.hdr   <-- ASCII Phoenix (Summary) header
 *
 *------------------------------------------------------------------------
 *
 * [Contacts]:
 *
 *   John F. Querns
 *   Veridian Engineering (Dayton Group)
 *   5200 Springfield Pike, Dayton, OH 45431
 *   email: jquerns@dytn.veridian.com
 *
 *   Veridian Contractor Area
 *   Area B  Bldg 23  Rm 115
 *   2010 Fifth Street
 *   Wright Patterson AFB, OH  45433-7001
 *   Work : (937) 255-1116, Ext 2818
 *   email: jquerns@mbvlab.wpafb.af.mil
 *
 *------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Define MSTAR image type */
#define CHIP_IMAGE 0
#define FSCENE_IMAGE 1

#define ALL_DATA 0
#define MAG_DATA 1

#define SUCCESS 0
#define FAILURE -1

#define LSB_FIRST 0 /* Implies little-endian CPU... */
#define MSB_FIRST 1 /* Implies big-endian CPU...    */

/* Function Declarations */
static float byteswap_SR_IR();
static unsigned short byteswap_SUS_IUS();
static int CheckByteOrder();

char* rindex(const char * s, int c)
{
    return strrchr(s, c);
}

void mstar2raw_main(argc, argv, PHXsize, PHXheader, CHIPsize, CHIPdata, FSCENEsize, FSCENEmag, FSCENEphase)
    int argc;
    char* argv[];
    int* PHXsize;
    char** PHXheader;
    long* CHIPsize;
    float** CHIPdata;
    long* FSCENEsize;
    unsigned short** FSCENEmag;
    unsigned short** FSCENEphase;
{

    /************************* D E C L A R A T I O N S *************************/

    FILE * MSTARfp = NULL; /* Input FILE ptr to MSTAR image file     */

    int i, n, numrows, numcols, numgot;

    char * MSTARname = NULL; /* Input MSTAR filename           */

    int phlen, nhlen, mstartype;
    long magloc, bytesPerImage, nchunks, totchunks;

    char* tptr = NULL; /* Temp buffer ptr */
    char* phdr = NULL; /* Ptr to buffer to hold Phoenix header */
    unsigned char tbuff[1024];

    unsigned short * FSCENEbuffer = NULL; /* Ptr to Fullscene data buffer */
    float * CHIPbuffer = NULL; /* Ptr to CHIp data buffer      */

    /* Byte Order Variables */
    int byteorder;
    unsigned char bigfloatbuf[4]; /* BigEndian float buffer... */
    float littlefloatval; /* LittleEndian float value  */
    unsigned char bigushortbuf[2]; /* BigEndian ushort buffer...*/
    unsigned short littleushortval; /* LittleEndian ushort value.*/

    /************************ B E G I N  C O D E ****************************/

    if (argc < 2)
    {
        fprintf(stderr, "\nUsage: mstar2raw <MSTAR Input>\n");
        exit(1);
    }
    else
    {
        MSTARname = argv[1];
    }

    /***************** MAIN MSTAR PROCESSING AREA ********************/

    MSTARfp = fopen(MSTARname, "rb");
    if (MSTARfp == NULL)
    {
        fprintf(stderr, "\n\nError: Unable to open [%s] for reading!\n\n", MSTARname);
        exit(1);
    }

    /****************************************************
     * Read first 512 bytes to figure out some header   *
     * parameters....                                   *
     ****************************************************/

    fread(tbuff, sizeof(char), 1024, MSTARfp);
    rewind(MSTARfp);

    /* Extract Phoenix Summary header length */
    tptr = strstr(tbuff, "PhoenixHeaderLength= ");
    if (tptr == (char*)NULL)
    {
        fprintf(stderr, "Can not determine Phoenix header length!\n");
        fclose(MSTARfp);
        exit(1);
    }
    else
    {
        sscanf(tptr + 20, "%d", & phlen);
    }

    /* Check for and extract native header length */
    tptr = strstr(tbuff, "native_header_length= ");
    if (tptr == (char*)NULL)
    {
        fprintf(stderr, "Can not determine native header length!\n");
        fclose(MSTARfp);
        exit(1);
    }
    else
    {
        sscanf(tptr + 21, "%d", & nhlen);
    }

    /* Extract MSTAR column width */
    tptr = strstr(tbuff, "NumberOfColumns= ");
    if (tptr == (char*)NULL)
    {
        fprintf(stderr, "Error: Can not determine MSTAR image width");
        fclose(MSTARfp);
        exit(1);
    }
    else
    {
        sscanf(tptr + 16, "%d", & numcols);
    }

    /* Extract MSTAR row height */
    tptr = strstr(tbuff, "NumberOfRows= ");
    if (tptr == (char*)NULL)
    {
        fprintf(stderr, "Error: Can not determine MSTAR image height!");
        fclose(MSTARfp);
        exit(1);
    }
    else
    {
        sscanf(tptr + 13, "%d", & numrows);
    }

    /* Set MSTAR image type */
    if (nhlen == 0)
    {
        /* Implies FLOAT MSTAR chip image */
        mstartype = CHIP_IMAGE;
    }
    else
    {
        mstartype = FSCENE_IMAGE; /* UnShort Fullscene */
    }

    /*******************************************************
     * Allocate memory to header buffer, read Phoenix hdr, *
     * and write out to output file...                     *
     *******************************************************/

    /* Allocate memory to Phoenix header buffer */
    phdr = (char*)malloc(phlen + 1);
    if (phdr == (char*)NULL)
    {
        fprintf(stderr, "Error: unable to allocate Phoenix header memory!\n");
        fclose(MSTARfp);
        exit(1);
    }

    /* Read Phoenix header into buffer */
    n = fread(phdr, sizeof(char), phlen, MSTARfp);
    if (n != phlen)
    {
        fprintf(stderr, "Error: in reading Phoenix header..only read [%d of %d] bytes\n", n, phlen);
        free(phdr);
        fclose(MSTARfp);
        exit(1);
    }

    /* Write Phoenix header to output header file */
    *PHXheader = malloc((phlen + 1) * sizeof(char));
    if (*PHXheader == NULL)
    {
        fprintf(stderr, "Error: unable to allocate output Phoenix header memory!\n");
        *PHXsize = 0;
    }
    else
    {
        memcpy(*PHXheader, phdr, (phlen + 1) * sizeof(char));
        *PHXsize = phlen;
    }

    /* Free Phoenix header memory...*/
    free(phdr);

    /******************************************************
     * Set up location to point to MSTAR magnitude data.. *
     ******************************************************/
    switch (mstartype)
    {
        case CHIP_IMAGE:
            magloc = phlen;
            fseek(MSTARfp, magloc, 0);
            break;

        case FSCENE_IMAGE:
            magloc = phlen + nhlen; /* nhlen = 512 */
            fseek(MSTARfp, magloc, 0);
            break;
    }
    nchunks = numrows * numcols;

    /******************************************************
     * Check byte-order, swap bytes if necessary...       *
     * Allocate memory, read data,  & convert to 8-bit    *
     * based on 'mstartype'                               *
     ******************************************************/

    /* Check byteorder */
    byteorder = CheckByteOrder();

    /******************************************************
     * Allocate memory, read data,  & write out based on  *
     * type of MSTAR image...and which data to write out  *
     *                                                    *
     * NOTE: For Chip data, I allocate all of the memory  *
     *       needed (magnitude+phase), read and then write*
     *       all of it out...                             *
     *                                                    *
     *       For fullscene data, because of the size of   *
     *       memory needed, I allocate only enough to     *
     *       grab the magnitude or the phase.  I then     *
     *       process first the magnitude and then the     *
     *       phase using the same buffer pointer....      *
     *                                                    *
     *       The code will read & write out ONLY the MAG  *
     *       image data if so specified by the user...    *
     ******************************************************/

    *CHIPsize = 0;
    *FSCENEsize = 0;

    switch (mstartype)
    {
        case CHIP_IMAGE:
        {
            totchunks = nchunks * 2;
            bytesPerImage = totchunks * sizeof(float);
            CHIPbuffer = (float*)malloc(bytesPerImage);
            if (CHIPbuffer == (float*)NULL)
            {
                fprintf(stderr, "Error: Unable to malloc CHIP memory!\n");
                fclose(MSTARfp);
                exit(1);
            }

            switch (byteorder)
            {
                case LSB_FIRST:
                    // Little-endian... do byteswap
                    for (i = 0; i < totchunks; i++)
                    {
                        fread(bigfloatbuf, sizeof(char), 4, MSTARfp);
                        littlefloatval = byteswap_SR_IR(bigfloatbuf);
                        CHIPbuffer[i] = littlefloatval;
                    }
                    break;

                case MSB_FIRST:
                    // Big-endian... no swap
                    numgot = fread(CHIPbuffer, sizeof(float), totchunks, MSTARfp);
                    break;
            }

            *CHIPdata = malloc(totchunks * sizeof(float));
            if (*CHIPdata == NULL)
            {
                fprintf(stderr, "Error: unable to allocate output CHIPdata memory!\n");
            }
            else
            {
                memcpy(*CHIPdata, CHIPbuffer, totchunks * sizeof(float));
                *CHIPsize = totchunks;
            }
            break; /* End of CHIP_IMAGE case */
        }

        case FSCENE_IMAGE:
        {
            bytesPerImage = nchunks * sizeof(short);
            FSCENEbuffer = (unsigned short*) malloc(bytesPerImage);
            if (FSCENEbuffer == (unsigned short*) NULL)
            {
                fprintf(stderr, "Error: Unable to malloc fullscene memory!\n");
                fclose(MSTARfp);
                exit(1);
            }

            switch (byteorder)
            {
                case LSB_FIRST:
                    // Little-endian... do byteswap
                    for (i = 0; i < nchunks; i++)
                    {
                        fread(bigushortbuf, sizeof(char), 2, MSTARfp);
                        littleushortval = byteswap_SUS_IUS(bigushortbuf);
                        FSCENEbuffer[i] = littleushortval;
                    }
                    break;

                case MSB_FIRST:
                    // Big-endian... no swap
                    numgot = fread(FSCENEbuffer, sizeof(short), nchunks, MSTARfp);
                    break;
            }

            *FSCENEmag = malloc(nchunks * sizeof(unsigned short));
            if (*FSCENEmag == NULL)
            {
                fprintf(stderr, "Error: unable to allocate output FSCENEmag memory!\n");
            }
            else
            {
                memcpy(*FSCENEmag, FSCENEbuffer, nchunks * sizeof(short));
                *FSCENEsize = nchunks;
            }

            switch (byteorder)
            {
                case LSB_FIRST:
                    // Little-endian... do byteswap
                    for (i = 0; i < nchunks; i++)
                    {
                        fread(bigushortbuf, sizeof(char), 2, MSTARfp);
                        littleushortval = byteswap_SUS_IUS(bigushortbuf);
                        FSCENEbuffer[i] = littleushortval;
                    }
                    break;

                case MSB_FIRST:
                    // Big-endian... no swap
                    numgot = fread(FSCENEbuffer, sizeof(short), nchunks, MSTARfp);
                    break;
            }

            *FSCENEphase = malloc(nchunks * sizeof(unsigned short));
            if (*FSCENEphase == NULL)
            {
                fprintf(stderr, "Error: unable to allocate output FSCENEphase memory!\n");
            }
            else
            {
                memcpy(*FSCENEphase, FSCENEbuffer, nchunks * sizeof(unsigned short));
                *FSCENEsize = nchunks;
            }

            /* Cleanup: free memory */
            free(FSCENEbuffer);

            break; /* End of FSCENE_IMAGE case */
        }
    } /* End of 'mstartype' switch */

    /* Cleanup: close files */
    fclose(MSTARfp);
}

/****************************** STATIC FUNCTIONS ******************************/

/************************************************
 * Function:    byteswap_SR_IR                  *
 *   Author:    Dave Hascher (Veridian Inc.)    *
 *     Date:    06/05/97                        *
 *    Email:    dhascher@dytn.veridian.com      *
 ************************************************
 * 'SR' --> Sun 32-bit float value              *
 * 'IR' --> PC-Intel 32-bit float value         *
 ************************************************/

static float byteswap_SR_IR(pointer)
    unsigned char *pointer;
{
    float *temp;
    unsigned char iarray[4], *charptr;

    iarray[0] = *(pointer + 3);
    iarray[1] = *(pointer + 2);
    iarray[2] = *(pointer + 1);
    iarray[3] = *pointer;
    charptr = iarray;
    temp = (float *)charptr;
    return *temp;
}


/************************************************
 * Function:    byteswap_SUS_IUS                *
 *   Author:    John Querns (Veridian Inc.)     *
 *     Date:    06/05/97                        *
 *    Email:    jquerns@dytn.veridian.com       *
 ************************************************
 * 'SUS' --> Sun 16-bit uns short value         *
 * 'IUS' --> PC-Intel 16-bit uns short value    *
 ************************************************/

static unsigned short byteswap_SUS_IUS(pointer)
    unsigned char *pointer;
{
    unsigned short *temp;
    unsigned char iarray[2], *charptr;

    iarray[0] = *(pointer + 1);
    iarray[1] = *pointer;
    charptr = iarray;
    temp = (unsigned short *)charptr;
    return *temp;
}


/**********************************
 *   checkByteOrder()             *
 **********************************
 * Taken from:                    *
 *                                *
 *   Encyclopedia of Graphic File *
 *   Formats, Murray & Van Ryper, *
 *   O'Reilly & Associates, 1994, *
 *   pp. 114-115.                 *
 *                                *
 * Desc: Checks byte-order of CPU.*
 **********************************/

static int CheckByteOrder(void)
{
    short w = 0x0001;
    char* b = (char *)&w;
    return b[0] ? LSB_FIRST : MSB_FIRST;
}

/************************** LAST LINE of mstar2raw.c **************************/