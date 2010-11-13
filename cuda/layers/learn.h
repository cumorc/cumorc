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
// methods that serve as the API
// header for learn.cu
// ==========================================================================

#ifndef _learn_h_
#define _learn_h_
void patchOfPyr(int npatches, char* _picname, char* _patchpattern);
void patchOfS1(int npatches, char* _picname, char* _gaborname, char* _patchpattern);
void patchOfC1(int npatches, char* _picname, char* _gaborfile, char* _patchpattern);
void testS2(int npatches, char* _picname, char* _gaborfile, char* _patchpattern);
void extract_patches(int npatches_per_pic, char* _gaborfile, char* _patchpattern, int randomseed);
void weight_patches(char* _gaborfile, char* _patchpattern, char* _weightfilename, char* _spheringfilename, bool training);
#endif
