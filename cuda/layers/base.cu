// ==========================================================================
// Version 1.0
// ==========================================================================
// (C)opyright: 2010
//
//   Ulm University
//
// Creator: Ferdinand Deger, <Firstname>.<Lastname>@gmail.com
// Creator: Helmut Sedding, <Firstname>@<Lastname>.net
// ==========================================================================
// this is the file which #includes anything else.
// is referenced from c-files that create separate binaries.
// ==========================================================================


#include <cstdio>
//#include <ctime>
#include <time.h>
#ifdef MACTIME
#include <sys/time.h>
#endif

struct timespec measure_ts_start, measure_ts_end;

// debug printf function
#define p(a,s) if(DEBUG) printf("%s %d\n",a,s);
#define pt(a,s) if(DEBUG) printf("%s %u\n",a,s);

#include "../util/io_tga.h"
#include "../util/io_largetga.h"
#include "../util/io_matlab.h"
// #include "util/head

//
#include "pyramid.cu"
#include "patches.cu"
#include "general.cu"
#include "pyramid_resize.cu"
#include "S1.cu"
#include "C1.cu"
#include "S2.cu"
#include "learn.cu"
#include "object_recognition.cu"
