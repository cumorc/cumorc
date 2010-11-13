#include "io_largetga.h"

#include <cstdio>
#include <cstring>

float* loadLargeTGAImage(const char* filename, unsigned int* resx, unsigned int* resy, unsigned int* numchannels) {
  unsigned char* data;
  struct tga_header {
      unsigned char id_len, pal_type, img_type;
      unsigned char pal_startH, pal_startL, pal_lenH, pal_lenL;
      unsigned char pal_entry_size;
      unsigned short x_0, y_0;
      unsigned int resx, resy;   // MOD for LARGE width/height
      unsigned char bpp, img_attribute;
  } head;

  
  FILE* f = fopen(filename, "rb");
  int io;
  if (f == 0) {fprintf(stderr, "\nERROR: could not load texture file %s", filename); return 0; }   
  
  io = fread(&head, 1, sizeof(struct tga_header), f);
  if(!io) {fprintf(stderr, "\nERROR: loading file %s", filename); return 0;}

  // bigendian for mac:
//    head.x_0 = _Swap16(head.x_0); head.y_0 = _Swap16(head.y_0);
//    head.resx = _Swap16(head.resx); head.resy = _Swap16(head.resy);
  
  if (head.img_type != 2 && head.img_type != 3) {fprintf(stderr, "\nERROR: unsupported TGA image type %d", head.img_type); return 0;}
  if (head.bpp != 24 && head.bpp != 32 && head.bpp != 8) {fprintf(stderr, "\nERROR: yet supported TGA bit depth %d.", head.bpp); return 0;}
  fseek(f, head.id_len, SEEK_CUR); // jump over the id
  
  *numchannels = head.bpp / 8;
  unsigned int size = head.resx * head.resy * (*numchannels);
  data = new unsigned char[size];
  io = fread(data, *numchannels, head.resx*head.resy, f);
  if(!io) {fprintf(stderr, "\nERROR: loading file %s", filename); return 0;}
  fclose(f);
  
  printf("Loaded Texture image %s (bpp: %d, %d %d)\n", filename, (int)head.bpp, head.resx, head.resy);
  *resx = head.resx; *resy = head.resy;

  // convert to float
  float* floatdata = new float[size];
  for(unsigned int i=size; i--; ) {
		floatdata[i] = (float)data[i] / 255.0f;
	//for(int x=*resx;x--;){
	//	for(int y=*resy;y--;){
	//		floatdata[*resx*y+x] = data[*resy*x+y] / 255.0f;
	//	}
  }

  delete[] data;

  return floatdata;
}

int saveLargeTGAImage(float* floatdata, const char* filename, unsigned int resx, unsigned int resy, unsigned int numchannels) {
  unsigned char* data;
  struct tga_header {
      unsigned char id_len, pal_type, img_type;
      unsigned char pal_startH, pal_startL, pal_lenH, pal_lenL;
      unsigned char pal_entry_size;
      unsigned short x_0, y_0;
      unsigned int resx, resy;
      unsigned char bpp, img_attribute;
  } head;
  memset(&head, 0, sizeof(struct tga_header));
  unsigned int size = resx * resy * numchannels;

  // convert to float
  data = new unsigned char[size];
  for(unsigned int i=size; i--; ) {
    data[i] = floatdata[i] * 255.0f;
  }
  
  head.img_type = 3;
  head.resx = resx;
  head.resy = resy;
  head.bpp = numchannels * 8;
  if (head.bpp != 24 && head.bpp != 32 && head.bpp != 8) {fprintf(stderr, "\nERROR: yet supported TGA bit depth %d.", head.bpp); return 0;}

  FILE* f = fopen(filename, "wb");
  int io;
  if (f == 0) {fprintf(stderr, "\nERROR: could not save texture file %s", filename); return 0; }   
  
  io = fwrite(&head, 1, sizeof(struct tga_header), f);
  if(!io) {fprintf(stderr, "\nERROR: saving file %s\n", filename); return 0;}
 
  if(size>0){
    io = fwrite(data, numchannels, resx*resy, f);
    if(!io) {fprintf(stderr, "\nERROR: saving file %s\n", filename); return 0;}
  }else {fprintf(stderr, "\nWARNING: saving empty file %s\n", filename);}
  fclose(f);
  
  printf("Saved Texture image %s (bpp: %d, %d %d)\n", filename, (int)head.bpp, head.resx, head.resy);
  delete[] data;
  return 1;
}

