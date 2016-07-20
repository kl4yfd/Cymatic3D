/*----------------------------------------------------------------------------
    Cymatic3D - real-time cymatics software

    Copyright (c) 2004 Ge Wang, Perry R. Cook, Ananya Misra.
        All rights reserved.
        http://soundlab.cs.princeton.edu/
        
   Copyright (c) 2014-2016 John A. Phelps
	https://github.com/kl4yfd/Cymatic3D
	kl4yfd@gmail.com

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
    U.S.A.
-----------------------------------------------------------------------------*/

//-----------------------------------------------------------------------------
// original name: sndpeek.cpp
// desc: small real-time spectrum visualizer, originally implemented 
//       by Ge Wang for the memex project and to entertain small children
//       at parties.
//
// authors: Ge Wang (gewang@cs.princeton.edu)
//          Perry R. Cook (prc@cs.princeton.edu)
//          Ananya Misra (amisra@cs.princeton.edu)
// thanks to:
//          Yilei Shao, Qin Lv, Tom Briggs
// library:
//          (STK) Perry R. Cook (prc@cs.princeton.edu)
//          (STK) Gary P. Scavone (gary@ccrma.stanford.edu)
//          (FFT) CARL CMusic Distribution
//          (Marsyas) George Tzanetakis (gtzan@cs.princeton.edu)
//          (libsndfile) Erik de Castro Lopo
// origin date: 11.28.2003 - ...
//
// usage: (type 'h' while running or see list of command line arguments)
//-----------------------------------------------------------------------------
#define SLEEPTIME 250

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <memory.h>

// libsndfile
#if defined(__USE_SNDFILE_NATIVE__)
#include <sndfile.h>
#else
#include "util_sndfile.h"
#endif

// STK
#include "RtAudio.h"
#include "Thread.h"

// OpenGL
#if defined(__OS_MACOSX__)
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
  #include <OpenGL/glu.h>
#else
  #include <GL/glut.h>
  #include <GL/gl.h>
  #include <GL/glu.h>
#endif

// process related
#if defined(__OS_WINDOWS__)
  #include <process.h>
  // usleep
  #ifndef usleep
  #define usleep(x) Sleep( (x/1000 <= 0 ? 1 : x/1000) )
  #endif
#else
  #include <unistd.h>
#endif

// FFT
#include "chuck_fft.h"

// Marsyas
#include "Centroid.h"
#include "DownSampler.h"
#include "Flux.h"
#include "LPC.h"
#include "MFCC.h"
#include "RMS.h"
#include "Rolloff.h"




//-----------------------------------------------------------------------------
// function prototypes
//-----------------------------------------------------------------------------
void idleFunc( );
void displayFunc( );
void reshapeFunc( int width, int height );
void keyboardFunc( unsigned char, int, int );
void keyboardUpFunc( unsigned char, int, int );
void specialFunc( int, int, int );
void mouseFunc( int button, int state, int x, int y );
void initialize_graphics( );
bool initialize_audio( );
void initialize_analysis( );
void extract_buffer( );
double compute_log_spacing( int fft_size, double factor );




//-----------------------------------------------------------------------------
// global variables and #defines
//-----------------------------------------------------------------------------
#define SAMPLE                  MY_FLOAT
#define SND_BUFFER_SIZE         1024
#define SND_FFT_SIZE            ( SND_BUFFER_SIZE * 2 )
#define SND_MARSYAS_SIZE        ( 512 )
#define INC_VAL_MOUSE           0.025f
#define INC_VAL_KB              0.025f


// width and height of the window
///
///
GLsizei g_width = 1024;
GLsizei g_height = 768;
GLsizei g_last_width = g_width;
GLsizei g_last_height = g_height;

// fill mode
GLenum g_fillmode = GL_POINTS;
// light 0 position
GLfloat g_light0_pos[4] = { 2.0f, 1.2f, 4.0f, 1.0f };
// light 1 parameters
GLfloat g_light1_ambient[] = { .2f, .2f, .2f, 1.0f };
GLfloat g_light1_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_light1_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_light1_pos[4] = { -2.0f, 0.0f, -4.0f, 1.0f };

// modelview stuff
GLfloat g_angle_x = 0.0f;
GLfloat g_angle_y = 0.0f;
GLfloat g_angle_z = 0.0f;
GLfloat g_x_inc = 0.0f;
GLfloat g_y_inc = 0.0f;
GLfloat g_z_inc = 120.0f;
GLfloat g_eye_x = 0.0f;
GLfloat g_eye_y = 0.0f; // 0.2f for stria, otherwise 0
GLfloat g_eye_z = 0.0f;
GLfloat g_eye_inc = 0.0f;
GLfloat g_zoom = 0.0f;

// Alpha
// Dual-eye 3D variables
GLfloat g_eye_separation = 0.05f;
GLfloat g_eye_left = 0.0f - g_eye_separation;
GLfloat g_eye_right = 0.0f + g_eye_separation;


// global audio buffer
SAMPLE g_fft_buffer[SND_FFT_SIZE];
SAMPLE g_audio_buffer[SND_BUFFER_SIZE]; // latest mono buffer (possibly preview)
SAMPLE g_stereo_buffer[SND_BUFFER_SIZE*2]; // current stereo buffer (now playing)
SAMPLE g_back_buffer[SND_BUFFER_SIZE]; // for lissajous
SAMPLE g_cur_buffer[SND_BUFFER_SIZE];  // current mono buffer (now playing), for lissajous
GLfloat g_window[SND_BUFFER_SIZE]; // DFT transform window
GLfloat g_log_positions[SND_FFT_SIZE/2]; // precompute positions for log spacing
GLint g_buffer_size = SND_BUFFER_SIZE;
GLint g_fft_size = SND_FFT_SIZE;

// Set default visualization
int visualization = 0;

// Set default rendertype
int rendertype = 0; // Dots / gl_points

bool liveshow = false;

// real-time audio
RtAudio * g_audio = NULL;
GLboolean g_ready = FALSE;
Mutex g_mutex;

// file reading
SNDFILE * g_sf = NULL;
SF_INFO g_sf_info;

// default sample rate
#if defined(__LINUX_ALSA__) || defined(__LINUX_OSS__) || defined(__LINUX_JACK__)
  GLuint g_srate = 48000;
#else
  GLuint g_srate = 44100;
#endif

// gain
GLfloat g_gain = 1.0f;
GLfloat g_time_scale = 1.0f;
GLfloat g_freq_scale = 1.0f;
///GLfloat g_lissajous_scale = 0.45f;
GLfloat g_lissajous_scale = 3.0f;

// how much to see
GLint g_time_view = 1;
GLint g_freq_view = 2;

// for log scaling
GLdouble g_log_space = 0;
GLdouble g_log_factor = 1;

// the index associated with the waterfall
GLint g_wf = 0;
// starting a file
GLint g_starting = 0;

// delay for pseudo-Lissajous in mono stuff
//GLint g_delay = SND_BUFFER_SIZE/2;
GLint g_delay = 100;

// number of real-time audio channels
GLint g_sndout = 0;
GLint g_sndin = 2;

// input filename
const char * g_filename = NULL;

// marsyas analysis modules
Centroid * g_centroid = NULL;
DownSampler * g_down_sampler = NULL;
Flux * g_flux = NULL;
LPC * g_lpc = NULL;
MFCC * g_mfcc = NULL;
RMS * g_rms = NULL;
Rolloff * g_rolloff = NULL;
Rolloff * g_rolloff2 = NULL;

/// Set some defaults and globals
// ---
// print features to stdout
GLboolean g_stdout = FALSE;
// opengl dislpay
GLboolean g_display = TRUE;
// fullscreen
GLboolean g_fullscreen = FALSE;
// waveform
GLboolean g_waveform = FALSE;
// lissajous
GLboolean g_lissajous = TRUE;
// plot waterfall
GLboolean g_wutrfall = FALSE;
// draw analysis features
GLboolean g_draw_features = FALSE;
// freeze display
GLboolean g_freeze = FALSE;
// Freeze-cycling
GLboolean g_freezecycle = TRUE;
GLboolean g_freezecycle_savestate = g_freezecycle;

GLboolean g_freezehold_savestate;

// mute audio
GLboolean g_mute = FALSE;
// use dB plot for spectrum
GLboolean g_usedb = FALSE;
// thing running
GLboolean g_running = TRUE;
// file input running
GLboolean g_file_running = FALSE;
// restart the file
GLboolean g_restart = TRUE;
// pause audio playback
GLboolean g_pause = FALSE;
// array of booleans for waterfall
GLboolean * g_draw = NULL;
// rainbowish coloring of waterfall?
GLboolean g_rainbow = TRUE;
// which way the waterfall moves
GLboolean g_backwards = FALSE;
// show time
GLboolean g_show_time = FALSE;
// draw play pointer
GLboolean g_draw_play = FALSE;
// freeze and hold until spacebar is released
GLboolean g_freezehold = FALSE;


// rotation increments, set to defaults
GLfloat g_inc_val_mouse = INC_VAL_MOUSE;
GLfloat g_inc_val_kb = INC_VAL_KB;

// shifting of zpos
GLfloat g_dz = .1f; 

// when to begin file reading
GLfloat g_begintime = 0; 

// not sure what these do anymore...
GLint g_buffer_count_a = 0;
GLint g_buffer_count_b = 0;

// for waterfall
struct Pt2D { float x; float y; };
Pt2D ** g_spectrums = NULL;
///GLuint g_depth = 48; // for john: 64
GLuint g_depth = 64; 
GLfloat g_z = 0.0f;
GLboolean g_z_set = FALSE;
GLfloat g_space = .12f; // for john: .1f
GLboolean g_downsample = FALSE;
GLint g_ds = 0; // downsample amount

// for time domain waterfall
SAMPLE ** g_waveforms = NULL;
GLfloat g_wf_delay_ratio = 1.0f / 3.0f;
GLuint g_wf_delay = (GLuint)(g_depth * g_wf_delay_ratio + .5f);
GLuint g_wf_index = 0;

// Set default
int freezecycle_skipframes = 4;


static float centroid_val, flux_val, rms_val, rolloff_val, rolloff2_val;

/// Set default background and foreground colors
int backgroundnumber = 0;
int chakranumber = 1;

/// 9-Chakra color system, which adds black and white (represents exessive grounding / un-grounding)
static const float ChakraColor[9][3] = {
	{0.0f, 0.0f, 0.0f},   // black
	{1.0f, 0.0f, 0.0f},   // red
	{1.0f, 0.64f, 0.0f},  // orange
	{1.0f, 1.0f, 0.0f},   // yellow
	{0.0f, 1.0f, 0.0f},   // green
	{0.0f, 0.0f, 1.0f},   // blue
	{0.29f, 0.0f, 0.51f}, // indigo
	{0.93f, 0.51f ,0.93f},// violet
	{1.0f, 1.0f, 1.0f}    // white
};


//-----------------------------------------------------------------------------
// name: help()
// desc: ...
//-----------------------------------------------------------------------------
void help()
{
    fprintf( stderr, "----------------------------------------------------\n" );
    fprintf( stderr, "== CYMATIC3D ==\n" );
    fprintf( stderr, " Copyright 2014-2016 John A. Phelps\n " );
    fprintf( stderr, " https://github.com/kl4yfd/Cymatic3D\n" );
    fprintf( stderr, "----------------------------------------------------\n" );
    fprintf( stderr, "'h' - print this help message\n" );
    fprintf( stderr, "'s' - print current settings to console\n" );
    fprintf( stderr, "'f' - toggle fullscreen\n" );
    fprintf( stderr, "'p' - pause frame (can still rotate/scale)\n" );
    //fprintf( stderr, "'1' - toggle waveform display\n" );
    //fprintf( stderr, "'2' - toggle lissajous display\n" );
    //fprintf( stderr, "'3' - (also 'w') toggle wutrfall display\n" );
    //fprintf( stderr, "'4' - toggle feature extraction (broken)\n" );
    //fprintf( stderr, "'d' - toggle dB plot for spectrum\n" );
    //fprintf( stderr, "'r' - toggle rainbow waterfall\n" );
    //fprintf( stderr, "'b' - toggle waterfall moving backwards/forwards\n" );
    //fprintf( stderr, "'x' - restart file playback (if applicable)\n" );
    //fprintf( stderr, "'m' - mute\n" );
    //fprintf( stderr, "'j' - move spectrum + z\n" );
    //fprintf( stderr, "'k' - move spectrum - z\n" );
    //fprintf( stderr, "'u' - spacing more\n" );
    //fprintf( stderr, "'i' - spacing less\n" );
    //fprintf( stderr, "'w' - scale DOWN the Waterfall\n" );
    //fprintf( stderr, "'W' - scale UP the Waterfall\n" );
    //fprintf( stderr, "'o' - scale DOWN the Oscope\n" );
    //fprintf( stderr, "'O' - scale UP the Oscope\n" );
    fprintf( stderr, "'-' - scale DOWN the Cymatic\n" );
    fprintf( stderr, "'+' - scale UP the Cymatic\n" );
    //fprintf( stderr, "'c' = LOWER log factor of spectrum display\n" );
    //fprintf( stderr, "'v' = RAISE log factor of spectrum display\n" );
    //fprintf( stderr, "      (log factor of 1.0 is linear display\n" );
    //fprintf( stderr, "(shift)'c' - LOWER amount of viewable waveform\n" );
    //fprintf( stderr, "(shift)'v' - RAISE amount of viewable waveform\n" );
    fprintf( stderr, "'y' - decrease delay for Cymatic\n" );
    fprintf( stderr, "'Y' - increase delay for Cymatic\n" );
    fprintf( stderr, "'Z' - toggle frame delay\n" );
    fprintf( stderr, "'1-8' - pre-programmed frame delays\n" );
    fprintf( stderr, "'9' - DEcrease frame delay\n" );
    fprintf( stderr, "'0' - INcrease frame delay\n" );


    /// TODO: More and better help()
    fprintf( stderr, "\n Up Arrow - rotate up\n" );
    fprintf( stderr, " Down Arrow  - rotate down\n" );
    fprintf( stderr, " Left Arrow - rotate left\n" );
    fprintf( stderr, " Right Arrow  - rotate right\n" );
    fprintf( stderr, " '['  - Roll Left\n" );
    fprintf( stderr, " ']'  - Roll Right\n" );
    fprintf( stderr, " '<'  - Previous Visualization\n" );
    fprintf( stderr, " '>'  - Next Visualization\n" );
    fprintf( stderr, " '/'  - Previous Render Type\n" );
    fprintf( stderr, " '?'  - Next Render Type\n" );    
    fprintf( stderr, " '|'  - Disable spin vortex\n" );    
    fprintf( stderr, " '\\'  - Enable spin vortex\n" );    
    fprintf( stderr, " 'HOME'  - Move camera back to default & disable all rotations / rolls.\n" );
    fprintf( stderr, " 'SPACEBAR'  - Hold to freeze the current view\n" );
    fprintf( stderr, " 'q' or 'ESC' - Quit\n" );
    fprintf( stderr, "----------------------------------------------------\n" );
    fprintf( stderr, "\n" );
}




//-----------------------------------------------------------------------------
// name: usage()
// desc: ...
//-----------------------------------------------------------------------------
void usage()
{
    fprintf( stderr, "usage: cymatic3d  --[options] [filename]\n" );
    fprintf( stderr, "  ON/OFF options: fullscreen|waveform|lissajous|waterfall|\n" );
    fprintf( stderr, "                  dB|features|fallcolors|backward|showtime|\n" );
    fprintf( stderr, "                  freeze|live\n" );
    fprintf( stderr, "  number options: timescale|freqscale|lissscale|logfactor|\n" );
    fprintf( stderr, "                  spacing|zpos|dzpos|depth|preview|yview|\n" );
    fprintf( stderr, "                  rotatem|rotatek|begintime|ds\n" );
    fprintf( stderr, "   other options: nodisplay|print\n" );
    fprintf( stderr, "\n" );
    fprintf( stderr, "example:\n" );
    fprintf( stderr, "    cymatic3d --live --fullscreen:ON \n" );
    fprintf( stderr, "\n" );
    fprintf( stderr, "\n" );
}




//-----------------------------------------------------------------------------
// Name: main( )
// Desc: entry point
//-----------------------------------------------------------------------------
int main( int argc, char ** argv )
{
    /// Set some environment variables that make the graphics look much better
  
	/// force V-Sync (super necessary)
	putenv( (char *) "__GL_SYNC_TO_VBLANK=1" );
	printf( "\nEnabling: Syncronize software to display refresh rate (VSYNC)" );
	
	/// TODO: test anisotropic
	putenv( (char *)  "__GL_LOG_MAX_ANISO=0" );
	//printf( "\nAnisotropic Filtering: 8X (mode 3)" );
	printf( "\nDisabling: Anisotropic Filtering: 1X (mode 0)" );

	// TODO: test  FSAA
	putenv( (char *)  " __GL_FSAA_MODE=4" );
	printf( "\nEnabling: Full Scene Antialiasing:\n\t 2x Supersampling -or-\n\t 4x Bilinear Multisampling\n\t (FSAA mode 4 differs per graphics card)" );
	
	printf( "\n" );

    GLboolean set_play = FALSE;

    // command line arguments
    for( int i = 1; i < argc; i++ )
    {
        if( !strncmp( argv[i], "-", 1 ) )
        {
            if( !strcmp( argv[i], "--print" ) )
                g_stdout = TRUE;
            else if( !strcmp( argv[i], "--help" ) || !strcmp( argv[i], "--about" ) )
            {
                usage();
                exit( 0 );
            }
            else if( !strcmp( argv[i], "--sndout" ) )
                g_sndout = 2;
	    else if( !strcmp( argv[i], "--live" ) )
                liveshow = true;
            else if( !strcmp( argv[i], "--nodisplay" ) )
                g_display = FALSE;
            else if( !strcmp(argv[i], "--fullscreen") || !strcmp(argv[i], "--fullscreen:ON") )
                g_fullscreen = TRUE;
            else if( !strcmp(argv[i], "--fullscreen:OFF") )
                g_fullscreen = FALSE;
            else if( !strcmp(argv[i], "--waveform") || !strcmp(argv[i], "--waveform:ON") )
                g_waveform = TRUE;
            else if( !strcmp(argv[i], "--waveform:OFF") )
                g_waveform = FALSE;
            else if( !strcmp(argv[i], "--lissajous") || !strcmp(argv[i], "--lissajous:ON") )
                g_lissajous = TRUE;
            else if( !strcmp(argv[i], "--lissajous:OFF") )
                g_lissajous = FALSE;
            else if( !strcmp(argv[i], "--waterfall") || !strcmp(argv[i], "--waterfall:ON") )
                g_wutrfall = TRUE;
            else if( !strcmp(argv[i], "--waterfall:OFF") )
                g_wutrfall = FALSE;
            else if( !strcmp(argv[i], "--dB") || !strcmp(argv[i], "--dB:ON") )
                g_usedb = TRUE;
            else if( !strcmp(argv[i], "--dB:OFF") )
                g_usedb = FALSE;
            else if( !strcmp(argv[i], "--features") || !strcmp(argv[i], "--features:ON") )
                g_draw_features = TRUE;
            else if( !strcmp(argv[i], "--features:OFF") )
                g_draw_features = FALSE;
            else if( !strcmp(argv[i], "--fallcolors") || !strcmp(argv[i], "--fallcolors:ON") )
                g_rainbow = TRUE;
            else if( !strcmp(argv[i], "--fallcolors:OFF") )
                g_rainbow = FALSE;
            else if( !strcmp(argv[i], "--backward") || !strcmp(argv[i], "--backward:ON") )
                g_backwards = TRUE;
            else if( !strcmp(argv[i], "--backward:OFF") )
                g_backwards = FALSE;
            else if( !strcmp(argv[i], "--showtime") || !strcmp(argv[i], "--showtime:ON") )
                g_show_time = TRUE;
            else if( !strcmp(argv[i], "--showtime:OFF") )
                g_show_time = FALSE; 
            else if( !strcmp(argv[i], "--freeze") || !strcmp(argv[i], "--freeze:ON") )
                g_freeze = g_pause = TRUE;
            else if( !strcmp(argv[i], "--freeze:OFF") )
                g_freeze = g_pause = FALSE; 
            else if( !strcmp(argv[i], "--drawplay") || !strcmp(argv[i], "--drawplay:ON") )
                g_draw_play = set_play = TRUE;
            else if( !strcmp(argv[i], "--drawplay:OFF") )
            {   g_show_time = FALSE; set_play = TRUE;   } 
            else if( !strncmp(argv[i], "--srate:", 8) )
                g_srate = atoi( argv[i]+8 ) > 0 ? atoi( argv[i]+8 ) : g_srate;
            else if( !strncmp(argv[i], "--timescale:", 12) )
                g_time_scale = atof( argv[i]+12 );
            else if( !strncmp(argv[i], "--freqscale:", 12) )
                g_freq_scale = atof( argv[i]+12 );
            else if( !strncmp(argv[i], "--lissscale:", 12) )
                g_lissajous_scale = atof( argv[i]+12 );
            else if( !strncmp(argv[i], "--logfactor:", 12) )
                g_log_factor = atof( argv[i]+12 ) > 0 ? atof( argv[i]+12 ) : g_log_factor;
            else if( !strncmp(argv[i], "--spacing:", 10) )
                g_space = atof( argv[i]+10 ) > 0 ? atof( argv[i]+10 ) : g_space;
            else if( !strncmp(argv[i], "--rotatem:", 10) )
                g_inc_val_mouse = atof( argv[i]+10 );
            else if( !strncmp(argv[i], "--rotatek:", 10) )
                g_inc_val_kb = atof( argv[i]+10 ) / (INC_VAL_MOUSE/INC_VAL_KB);
            else if( !strncmp(argv[i], "--zpos:", 7 ) )
            {
                g_z = atof( argv[i]+7);
                g_z_set = TRUE;
            }
            else if( !strncmp(argv[i], "--dzpos:", 8 ) )
                g_dz = atof( argv[i]+8 ); 
            else if( !strncmp(argv[i], "--depth:", 8) )
            {
                g_depth = atoi( argv[i]+8 ) > 0 ? atoi( argv[i]+8 ) : g_depth;
            }
            else if( !strncmp(argv[i], "--preview:", 10) )
            {
                g_wf_delay_ratio = atof( argv[i]+10 );
                if( g_wf_delay_ratio < 0 || g_wf_delay_ratio >= 1 )
                {
                    fprintf( stderr, "[Cymatic3d]: --preview requires 0 <= value < 1...\n" );
                    usage();
                    return -1;
                }
            }
            else if( !strncmp(argv[i], "--yview:", 8) )
                g_eye_y = atof( argv[i] + 8 );
            else if( !strncmp( argv[i], "--begintime:", 12 ) )
                g_begintime = atof( argv[i]+12 ) >= 0 ? atof( argv[i]+12 ) : g_begintime; 
            else if( !strncmp( argv[i], "--ds:", 5 ) )
            {
                g_ds = atoi( argv[i] + 5 ) >= 0 ? atoi( argv[i] + 5 ) : 0; 
                g_downsample = (g_ds > 0); 
            }
            else
            {
                fprintf( stderr, "[Cymatic3d]: unrecognized option '%s'...\n", argv[i] );
                usage();
                return -1;
            }
        }
        else
        {
            if( g_filename )
            {
                fprintf( stderr, "[Cymatic3d]: multiple filenames specified...\n" );
                usage();
                return -2;
            }
            
            // reading from file...
            g_filename = argv[i];
            g_sndout = 2;
            g_starting = 1;
        }
    }

    // compute delay, but disable delay if it's mic input
    g_wf_delay = g_filename ? (GLuint)(g_wf_delay_ratio * g_depth + .5f) : 0;

    // infer settings
    if( g_filename ) g_sndin = 0;
    if( !g_sndin && !g_sndout ) g_display = FALSE;
    if( !set_play && g_filename ) g_draw_play = TRUE;

    // if using graphics
    if( g_display )
    {
#ifdef __OS_MACOSX__
        //save working dir
        char * cwd = getcwd( NULL, 0 );
#endif
        // initialize GLUT
        glutInit( &argc, argv );

#ifdef __OS_MACOSX__
        //restore working dir
        chdir( cwd );
        free( cwd );
#endif

        // double buffer, use rgb color, enable depth buffer
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
        // initialize the window size
        glutInitWindowSize( g_width, g_height );
        // set the window postion
        glutInitWindowPosition( 100, 100 );
        // create the window
        glutCreateWindow( "Cymatic3d: The Sacred Geometry of Sound " );
        // full screen
        if( g_fullscreen )
            glutFullScreen();

        // set the idle function - called when idle
        glutIdleFunc( idleFunc );
        // set the display function - called when redrawing
        glutDisplayFunc( displayFunc );
        // set the reshape function - called when client area changes
        glutReshapeFunc( reshapeFunc );
        // set the keyboard function - called on keyboard key-down/pressed events
        glutKeyboardFunc( keyboardFunc );
	// set the keyboard function - called on keyboard events
        glutSpecialFunc( specialFunc );
        // set the mouse function - called on mouse stuff
        glutMouseFunc( mouseFunc );
        // set the other keyboard function - called on keyboard key-up/released events
        glutKeyboardUpFunc( keyboardUpFunc );
        
        // do our own initialization
        initialize_graphics( );
    }
    
    // initialize analysis
    initialize_analysis( );
    
    // intialize real-time audio
    if( !initialize_audio( ) )
    {
        fprintf( stderr, "[Cymatic3d]: exiting...\n" );
        return -3;
    }

    // display mode
    if( g_display )
    {
        // let GLUT handle the current thread from here
        glutMainLoop();
    }
    else
    {
        // no graphics... run our own main loop
        sf_count_t count;
        while( g_running )
        {
            if( g_filename && !g_sndout )
            {
                // TODO: (to check) does this work for stereo files? 
                count = sf_read_float( g_sf, g_audio_buffer, g_buffer_size );
                g_buffer_count_a++;
                g_ready = TRUE;
                if( !count )
                {
                    g_file_running = FALSE;
                    fprintf( stderr, "[Cymatic3d]: file not running...\n" );
                }
            }

            // extract features
            extract_buffer();
        }
    }

    return 0;
}




//-----------------------------------------------------------------------------
// name: cb()
// desc: audio callback
//-----------------------------------------------------------------------------
int cb( char * buffer, int buffer_size, void * user_data )
{
    // freeze frame
    if( g_freeze ) {
        memset( buffer, 0, buffer_size * 2 * sizeof(SAMPLE) );
        g_ready = TRUE;
        return 0;
    }

    // lock
    g_mutex.lock();
    // check if reading from file
    if( !g_filename )
    {
        // copy
        memcpy( g_stereo_buffer, buffer, buffer_size * 2 * sizeof(SAMPLE) );
        // convert stereo to mono
        for( int i = 0; i < buffer_size; i++)
        {
            g_audio_buffer[i] = g_stereo_buffer[i*2] + g_stereo_buffer[i*2+1];
            g_audio_buffer[i] /= 2.0f;
        }
    }
    else
    {
        // check for restart
        if( g_restart )
        {
            // set playback position to begin
            sf_seek( g_sf, (int)(g_begintime * g_srate), SEEK_SET );
            g_wf_index = 0;
            g_wf = 0;
            g_starting = 1;
            g_restart = FALSE;
            // clear waveforms and waterfall and drawing booleans
            for( GLint i = 0; i < g_wf_delay || i < g_depth; i++ )
            {
                if( i < g_wf_delay )
                    memset( g_waveforms[i], 0, g_buffer_size * 2 * sizeof(SAMPLE) );
                if( i < g_depth )
                {
                    memset( g_spectrums[i], 0, sizeof(Pt2D)*SND_FFT_SIZE );
                    g_draw[i] = false;
                }
            }
        }
        
        // if not done yet...
        if( sf_seek(g_sf, 0, SEEK_CUR) < g_sf_info.frames && !g_pause )
        {
            // get the mono/stereo version
            sf_readf_float( g_sf, g_stereo_buffer, buffer_size );

            // if stereo, convert to mono
            if( g_sf_info.channels == 2 )
            {
                // convert stereo to mono
                for( int i = 0; i < buffer_size; i++)
                {
                    g_audio_buffer[i] = g_stereo_buffer[i*2] + g_stereo_buffer[i*2+1];
                    g_audio_buffer[i] /= 2.0f;
                }
            }
            else
            {
                // actually mono
                memcpy( g_audio_buffer, g_stereo_buffer, buffer_size * sizeof(SAMPLE) );
                // convert mono to stereo
                for( int i = 0; i < buffer_size; i++ )
                {
                    g_stereo_buffer[i*2] = g_stereo_buffer[i*2+1] = g_audio_buffer[i];
                }
            }

            // time-domain waterfall delay
            if( g_waveforms != NULL )
            {
                // put current buffer in time-domain waterfall
                memcpy( g_waveforms[g_wf_index], g_stereo_buffer, buffer_size * 2 * sizeof(SAMPLE) );
                // incrment index (this is also the index to copy out of)
                g_wf_index = (g_wf_index + 1) % g_wf_delay;
                // copy delayed buffer out of time-domain waterfall
                memcpy( g_stereo_buffer, g_waveforms[g_wf_index], buffer_size * 2 * sizeof(SAMPLE) );
            }

            // play stereo
            memcpy( buffer, g_stereo_buffer, buffer_size * 2 * sizeof(SAMPLE) );

            // count
            g_buffer_count_a++;
        }
        else
        {
            // done...
            g_running = FALSE;
            // zero
            memset( g_audio_buffer, 0, buffer_size * sizeof(SAMPLE) );
            // copy remaining delayed waveform buffers one by one
            if( g_wf_delay )
            {
                memset( g_waveforms[g_wf_index], 0, buffer_size * 2 * sizeof(SAMPLE) );
                g_wf_index = (g_wf_index + 1) % g_wf_delay; 
                memcpy( g_stereo_buffer, g_waveforms[g_wf_index], buffer_size * 2 * sizeof(SAMPLE) );
                memcpy( buffer, g_stereo_buffer, buffer_size * 2 * sizeof(SAMPLE) );
            }
            else
                memset( buffer, 0, 2 * buffer_size * sizeof(SAMPLE) );
        }
    }
    
    // set flag
    g_ready = TRUE;
    // unlock
    g_mutex.unlock();

    // mute the real-time audio
    if( g_mute )
        memset( buffer, 0, buffer_size * 2 * sizeof(SAMPLE) );

    return 0;
}




//-----------------------------------------------------------------------------
// Name: initialize_audio( )
// Desc: set up audio capture and playback and initializes any application data
//-----------------------------------------------------------------------------
bool initialize_audio( )
{
    // set sample rate
    Stk::setSampleRate( g_srate );
    
    // read from file
    if( g_filename )
    {
        fprintf( stderr, "[Cymatic3d]: opening %s...\n", g_filename );
        // attempt to open file
        g_sf = sf_open( g_filename, SFM_READ, &g_sf_info );
        if( !g_sf )
        {
            // exception
            fprintf( stderr, "[Cymatic3d]: error: cannot open '%s'...\n", g_filename );
            return false;
        }

        // set flag        
        g_file_running = TRUE;

        // set srate from the WvIn
        fprintf( stderr, "[Cymatic3d]: setting sample rate to %d\n", g_srate );
        g_srate = g_sf_info.samplerate;
    }
    else
    {
        // no time-domain waterfall delay!
        g_wf_delay = 0;
    }

    // make sound
    if( !g_filename || g_sndout )
    {
        // initialize rtaudio
        try
        {
            // buffer size
            int bufsize = g_buffer_size;
            // open the audio device for capture and playback
            g_audio = new RtAudio( 0, g_sndout, 0, g_sndin, RTAUDIO_FLOAT32,
                g_srate, &bufsize, 8 );
            // test
            if( bufsize != g_buffer_size )
            {
                // potential problem
                fprintf( stderr, "[Cymatic3d]: warning: using different buffer sizes: %i : %i\n",
                    bufsize, g_buffer_size );
            }
        }
        catch( StkError & e )
        {
            // exception
            fprintf( stderr, "[Cymatic3d](via RtAudio): %s\n", e.getMessage() );
            fprintf( stderr, "[Cymatic3d]: error: cannot open audio device for capture/playback...\n" );
            return false;
        }

        // set the audio callback
        g_audio->setStreamCallback( cb, NULL );
    
        // start the audio
        g_audio->startStream( );
    }

    // make the transform window
    hanning( g_window, g_buffer_size );
    
    // initialize
    if( g_wf_delay )
    {
        g_waveforms = new SAMPLE *[g_wf_delay];
        for( int i = 0; i < g_wf_delay; i++ )
        {
            // allocate memory (stereo)
            g_waveforms[i] = new SAMPLE[g_buffer_size * 2];
            // zero it
            memset( g_waveforms[i], 0, g_buffer_size * 2 * sizeof(SAMPLE) );
        }
    }

    return true;
}




//-----------------------------------------------------------------------------
// Name: initialize_analysis( )
// Desc: sets initial audio analysis parameters
//-----------------------------------------------------------------------------
void initialize_analysis( )
{
    // spectral centroid
    g_centroid = new Centroid( SND_MARSYAS_SIZE );
    // down sampler
    g_down_sampler = new DownSampler( g_buffer_size, 2 );
    // flux
    g_flux = new Flux( SND_MARSYAS_SIZE );
    // lpc
    g_lpc = new LPC( SND_MARSYAS_SIZE );
    g_lpc->init();
    // mfcc
    g_mfcc = new MFCC( SND_MARSYAS_SIZE, 0 );
    g_mfcc->init();
    // rms
    g_rms = new RMS( SND_MARSYAS_SIZE );
    // rolloff
    g_rolloff = new Rolloff( SND_MARSYAS_SIZE, 0.5f );
    g_rolloff2 = new Rolloff( SND_MARSYAS_SIZE, 0.8f );
}




//-----------------------------------------------------------------------------
// Name: initialize_graphics( )
// Desc: sets initial OpenGL states and initializes any application data
//-----------------------------------------------------------------------------
void initialize_graphics()
{
    // set the GL clear color - use when the color buffer is cleared
    glClearColor( 0.0f, 0.0f,0.0f, 1.0f );
    // set the shading model to 'smooth'
    glShadeModel( GL_SMOOTH );
    // enable depth
    glEnable( GL_DEPTH_TEST );
    // set the front faces of polygons
    glFrontFace( GL_CCW );
    // set fill mode
    glPolygonMode( GL_FRONT_AND_BACK, g_fillmode );
    // enable lighting
    glEnable( GL_LIGHTING );
    // enable lighting for front
    glLightModeli( GL_FRONT_AND_BACK, GL_TRUE );
    // material have diffuse and ambient lighting 
    glColorMaterial( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
    // enable color
    glEnable( GL_COLOR_MATERIAL );

    // enable light 0
    glEnable( GL_LIGHT0 );

    // setup and enable light 1
    glLightfv( GL_LIGHT1, GL_AMBIENT, g_light1_ambient );
    glLightfv( GL_LIGHT1, GL_DIFFUSE, g_light1_diffuse );
    glLightfv( GL_LIGHT1, GL_SPECULAR, g_light1_specular );
    glEnable( GL_LIGHT1 );
 
    // blend? (Jeff's contribution)
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    //glBlendFunc( GL_ONE, GL_DST_COLOR );
    glBlendEquation(GL_FUNC_ADD);
    glEnable( GL_BLEND );

    // initialize
    g_spectrums = new Pt2D *[g_depth];
    for( int i = 0; i < g_depth; i++ )
    {
        g_spectrums[i] = new Pt2D[SND_FFT_SIZE];
        memset( g_spectrums[i], 0, sizeof(Pt2D)*SND_FFT_SIZE );
    }
    g_draw = new GLboolean[g_depth];
    memset( g_draw, 0, sizeof(GLboolean)*g_depth );

    // compute log spacing
    g_log_space = compute_log_spacing( g_fft_size / 2, g_log_factor );

    help();
}




//-----------------------------------------------------------------------------
// Name: reshapeFunc( )
// Desc: called when window size changes
//-----------------------------------------------------------------------------
void reshapeFunc( int w, int h )
{
    // save the new window size
    g_width = w; g_height = h;
    // map the view port to the client area
    glViewport( 0, 0, w, h );
    // set the matrix mode to project
    glMatrixMode( GL_PROJECTION );
    // load the identity matrix
    glLoadIdentity( );
    // create the viewing frustum
    gluPerspective( 45.0, (GLfloat) w / (GLfloat) h, 1.0, 300.0 );
    // set the matrix mode to modelview
    glMatrixMode( GL_MODELVIEW );
    // load the identity matrix
    glLoadIdentity( );
    // position the view point
    gluLookAt( 0.0f, 3.5f * sin( g_eye_y ), 3.5f * cos( g_eye_y ), 
               0.0f, 0.0f, 0.0f, 
               0.0f, ( cos( g_eye_y ) < 0 ? -1.0f : 1.0f ), 0.0f );

    // set the position of the lights
    glLightfv( GL_LIGHT0, GL_POSITION, g_light0_pos );
    glLightfv( GL_LIGHT1, GL_POSITION, g_light1_pos );
    
    /* kl4yfd
     * Super-experimental Anaglyph 3D code 
     * 
     * 
     */
    
}

//-----------------------------------------------------------------------------
// Name: specialFunc( )
// Desc: key event
//-----------------------------------------------------------------------------
void specialFunc( int key, int x, int y )
{
    switch( key )
    {
      case GLUT_KEY_LEFT :
        g_y_inc -= g_inc_val_mouse;
	break;
      case GLUT_KEY_RIGHT :
        g_y_inc += g_inc_val_mouse;
	break;
	
      case GLUT_KEY_UP:
        g_x_inc += g_inc_val_kb;
        fprintf( stderr, "[Cymatic3d]: yview:%f\n", g_eye_y );
    break;
    case GLUT_KEY_DOWN:
        g_x_inc -= g_inc_val_kb;
        fprintf( stderr, "[Cymatic3d]: yview:%f\n", g_eye_y );
    break;
    
      case GLUT_KEY_HOME :
        g_x_inc = 0.0f;
        g_y_inc = 0.0f;
        ///g_z_inc = 0.0f;
	g_angle_x = 0.0f;
	g_angle_y = 0.0f;
	g_angle_z = 0.0f;
	g_eye_y = 0.0f;
	g_eye_inc = 0.0f;
	g_zoom = 0.0f;
	break;
	
      // TODO: map function keys to something useful
      /*
      case GLUT_KEY_F1 :
	break;
	      
      case GLUT_KEY_F2 :
	break;
	      
      case GLUT_KEY_F3 :
	break;
	  
      case GLUT_KEY_F4 :
	break;
	
      case GLUT_KEY_F5 :
	break;

      case GLUT_KEY_F6 :
	break;
	
      case GLUT_KEY_F7 :
	break;

      case GLUT_KEY_F8 :
	break;
	
      case GLUT_KEY_F9 :
	break;

      case GLUT_KEY_F10 :
	break;
	
      case GLUT_KEY_F11 :
	break;

      case GLUT_KEY_F12 :
	break;
      */
	
      //default:
	//g_inc = 0.0f;
    }
    
    
    // do a reshape since g_eye_y might have changed
    reshapeFunc( g_width, g_height );
    glutPostRedisplay( );
    
}


//-----------------------------------------------------------------------------
// Name: keyboardFunc( )
// Desc: key event keypress
//-----------------------------------------------------------------------------
void keyboardUpFunc( unsigned char key, int x, int y )
{
  	if ( key == 32 ) {
		g_freeze = g_freezehold_savestate;
		g_freezehold = false;
	}
  
}

//-----------------------------------------------------------------------------
// Name: keyboardFunc( )
// Desc: key event keypress
//-----------------------------------------------------------------------------
void keyboardFunc( unsigned char key, int x, int y )
{
    switch( key )
    {
    case '[':
        g_z_inc += g_inc_val_kb;
        fprintf( stderr, "[Cymatic3d]: zview:%f\n", g_z_inc );
	break;
    case ']':
        g_z_inc -= g_inc_val_kb;
        fprintf( stderr, "[Cymatic3d]: zview:%f\n", g_z_inc );
	break;    
    case '|':
        g_z_inc = 0;
        fprintf( stderr, "[Cymatic3d]: zview:%f\n", g_z_inc );
	break;
    case '\\':
        g_z_inc = 120;
        fprintf( stderr, "[Cymatic3d]: zview:%f\n", g_z_inc );
	break;
    case '{':
        g_zoom -= g_inc_val_kb;
        fprintf( stderr, "[Cymatic3d]: zview:%f\n", g_zoom );
	break;
    case '}':
        g_zoom += g_inc_val_kb;
        fprintf( stderr, "[Cymatic3d]: zview:%f\n", g_zoom );
	break;
    case '>':
    case '.':
        visualization++;
	if ( visualization > 30) visualization = 0;
        fprintf( stderr, "[Cymatic3d]: visualization:%d\n", visualization );
	break;
    case '<':
    case ',':
        visualization--;
	if ( visualization < 0) visualization = 0;
        fprintf( stderr, "[Cymatic3d]: visualization:%d\n", visualization );
	break;
    case '/':
        rendertype--;
	if ( rendertype < 0) rendertype = 0;
        fprintf( stderr, "[Cymatic3d]: rendertype:%d\n", rendertype );
	break;
    case '?':
        rendertype++;
	if ( rendertype > 8) rendertype = 0;
        fprintf( stderr, "[Cymatic3d]: rendertype:%d\n", rendertype );
	break;
    case 'j':
        g_z += g_dz;
        fprintf( stderr, "[Cymatic3d]: zpos:%f\n", g_z );
    break;
    case 'k':
        g_z -= g_dz;
        fprintf( stderr, "[Cymatic3d]: zpos:%f\n", g_z );
    break;
    case 'u':
        g_space *= 1.02f;
        fprintf( stderr, "[Cymatic3d]: spacing:%f\n", g_space );
    break;
    case 'i':
        g_space *= .98f;
        fprintf( stderr, "[Cymatic3d]: spacing:%f\n", g_space );
    break;
    /* 
    case '1':
        g_waveform = !g_waveform;
        fprintf( stderr, "[Cymatic3d]: waveform:%s\n", g_waveform ? "ON" : "OFF" );
    break;
    case '3':
        g_wutrfall = !g_wutrfall;
        fprintf( stderr, "[Cymatic3d]: waterfall:%s\n", g_wutrfall ? "ON" : "OFF" );
    break;
    */
    case 'd':
        g_usedb = !g_usedb;
        fprintf( stderr, "[Cymatic3d]: dB:%s\n", g_usedb ? "ON" : "OFF" );
    break;
    
    case 'a':
        g_draw_features = !g_draw_features;
        fprintf( stderr, "[Cymatic3d]: features:%s\n", g_draw_features ? "ON" : "OFF" );
    break;
    
    case 'q':
    case 27: // ESC KEY
        if (liveshow) fprintf( stderr, "\n LIVE SHOW MODE: QUITTING IS NOT AN OPTION! \n");
	else exit( 0 );
    break;
    case 'o':
        g_time_scale *= .99f;
        fprintf( stderr, "[Cymatic3d]: Oscope Scale:%f\n", g_time_scale );
    break;
    case 'O':
        g_time_scale *= 1.01f;
        fprintf( stderr, "[Cymatic3d]: Oscope Scale:%f\n", g_time_scale ); 
    break;
    case 'w':
        g_freq_scale *= .99f;
        fprintf( stderr, "[Cymatic3d]: Waterfall Scale:%f\n", g_freq_scale );
    break;
    case 'W':
        g_freq_scale *= 1.01f;
        fprintf( stderr, "[Cymatic3d]: Waterfall Scale:%f\n", g_freq_scale );
    break;
    
    case 'V':
        if( g_time_view > 1 )
            g_time_view--;

        fprintf( stderr, "[Cymatic3d]: time domain %i samples", g_buffer_size / g_time_view );
        fprintf( stderr, g_time_view == 1 ? " - (MAX)\n" : "\n" );
    break;
/// Old code. Re-use key elsewhere    
/*
    case 'C':
        if( g_time_view < 32 )
            g_time_view++;

        fprintf( stderr, "[Cymatic3d]: time domain %i samples", g_buffer_size / g_time_view );
        fprintf( stderr, g_time_view == 32 ? " - (MIN)\n" : "\n" );
    break;
*/
    case 'h':
    case 'H':
        help();
    break;
    case 'f':
    case 'F':
    {
        if( !g_fullscreen )
        {
            g_last_width = g_width;
            g_last_height = g_height;
            glutFullScreen();
        }
        else
            glutReshapeWindow( g_last_width, g_last_height );

        g_fullscreen = !g_fullscreen;
        fprintf( stderr, "[Cymatic3d]: fullscreen:%s\n", g_fullscreen ? "ON" : "OFF" );
    }
    break;
    case 'm':
    case 'M':
        g_mute = !g_mute;
        fprintf( stderr, "[Cymatic3d]: mute:%s\n", g_mute ? "ON" : "OFF" );
    break;
    case 'x':
        if( g_sf )
        {
            g_restart = TRUE;
            fprintf( stderr, "[Cymatic3d]: restarting file...\n" );
        }
    break;
    /*
    case '2':
        g_lissajous = !g_lissajous;
        fprintf( stderr, "[Cymatic3d]: lissajous:%s\n", g_lissajous ? "ON" : "OFF" );
    break;
    */
    case '-':
        g_lissajous_scale *= .95f;
        fprintf( stderr, "[Cymatic3d]: Cymatic Scale:%f\n", g_lissajous_scale );
    break;
    case '=':
        g_lissajous_scale *= 1.05f;
        fprintf( stderr, "[Cymatic3d]: Cymatic Scale:%f\n", g_lissajous_scale );
    break;
    case 'y':
        g_delay -= 1;
        if( g_delay < 0 )
            g_delay = 1;
        fprintf( stderr, "[Cymatic3d]: lissdelay = %i\n", g_delay );
    break;
    case 'Y':
        g_delay += 1;
        if( g_delay > g_buffer_size )
            g_delay = g_buffer_size;
        fprintf( stderr, "[Cymatic3d]: lissdelay = %i\n", g_delay );
    break;
    case 'p':
    case 'P':
	if (g_pause == false) {
		g_freeze = g_pause = true;
		g_freezecycle_savestate = g_freezecycle;
		g_freezecycle = false;
		fprintf( stderr, "[Cymatic3d]: Paused\n" );
	} else {
		g_freeze = g_pause = false;
		g_freezecycle = g_freezecycle_savestate;
	}
    break;
    case 'z':
    case 'Z':
      	if (g_freezecycle == true) {
		g_freezecycle = g_freeze = false;
		fprintf( stderr, "[Cymatic3d]: Frame Delay: OFF\n" );
	} else {
		g_freezecycle = !g_freezecycle;
		fprintf( stderr, "[Cymatic3d]: Frame Delay: ON\n" );
	}
    break;
    case '1':
	freezecycle_skipframes = 1;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case '2':
	freezecycle_skipframes = 2;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case '3':
	freezecycle_skipframes = 3;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case '4':
	freezecycle_skipframes = 4;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case '5':
	freezecycle_skipframes = 5;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case '6':
	freezecycle_skipframes = 6;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case '7':
	freezecycle_skipframes = 22;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case '8':
	freezecycle_skipframes = 42;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case '9':
	freezecycle_skipframes -= 2;
	if (freezecycle_skipframes < 0) freezecycle_skipframes = 0;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case '0':
	freezecycle_skipframes += 2;
	if (freezecycle_skipframes > 1000) freezecycle_skipframes = 1000;
        fprintf( stderr, "[Cymatic3d]: Frame Delay:%d\n", freezecycle_skipframes );
    break;
    case 'v':
        g_log_factor *= .98; //.99985;
        g_log_space = compute_log_spacing( g_fft_size / 2, g_log_factor );
        fprintf( stderr, "[Cymatic3d]: logfactor:%f\n", g_log_factor );
    break;
/// Old code, re-use 'c' elsewhere
/*    
    case 'c':
        g_log_factor /= .98; //.99985;
        g_log_space = compute_log_spacing( g_fft_size / 2, g_log_factor );
        fprintf( stderr, "[Cymatic3d]: logfactor:%f\n", g_log_factor );
    break;
*/
    case 'r':
        g_rainbow = !g_rainbow;
        fprintf( stderr, "[Cymatic3d]: fallcolors:%s\n", g_rainbow ? "ON" : "OFF" );
    break;
    case 't':
        g_show_time = !g_show_time; 
        fprintf( stderr, "[Cymatic3d]: show time:%s\n", g_show_time ? "ON" : "OFF" ); 
    break;
/*
    case 'b':
        g_backwards = !g_backwards;
        fprintf( stderr, "[Cymatic3d]: backward:%s\n", g_backwards ? "ON" : "OFF" );
    break;
*/
    case 'C':
    case 'c':
	chakranumber += 1;
	if (chakranumber > 8) chakranumber = 0;
        fprintf( stderr, "[Cymatic3d]: chakranumber:%d\n", chakranumber );
    break;    
    case 'B':
    case 'b':
	backgroundnumber += 1;
	if (backgroundnumber > 8) backgroundnumber = 0;
	fprintf( stderr, "[Cymatic3d]: backgroundnumber:%d\n", backgroundnumber );
    break;

    case 's':
        fprintf( stderr, "----------------------------------------------------\n" );
        fprintf( stderr, " current settings...\n" );
        fprintf( stderr, "----------------------------------------------------\n" );
        fprintf( stderr, "[Cymatic3d]: waveform:%s\n", g_waveform ? "ON" : "OFF" );
        fprintf( stderr, "[Cymatic3d]: lissajous:%s\n", g_lissajous ? "ON" : "OFF" );
        fprintf( stderr, "[Cymatic3d]: waterfall:%s\n", g_wutrfall ? "ON" : "OFF" );
        fprintf( stderr, "[Cymatic3d]: features:%s\n", g_draw_features ? "ON" : "OFF" );
        fprintf( stderr, "[Cymatic3d]: fallcolors:%s\n", g_rainbow ? "ON" : "OFF" );
        fprintf( stderr, "[Cymatic3d]: backward:%s\n", g_backwards ? "ON" : "OFF" );
        fprintf( stderr, "[Cymatic3d]: fullscreen:%s\n", g_fullscreen ? "ON" : "OFF" );
        fprintf( stderr, "[Cymatic3d]: dB:%s\n", g_usedb ? "ON" : "OFF" );
        fprintf( stderr, "[Cymatic3d]: mute:%s\n", g_mute ? "ON" : "OFF" );
        fprintf( stderr, "[Cymatic3d]: showtime:%s\n", g_show_time ? "ON" : "OFF" ); 
        fprintf( stderr, "[Cymatic3d]: freeze:%s\n", g_freeze ? "ON" : "OFF" ); 
        fprintf( stderr, "[Cymatic3d]: timescale:%f\n", g_time_scale ); 
        fprintf( stderr, "[Cymatic3d]: freqscale:%f\n", g_freq_scale );
        fprintf( stderr, "[Cymatic3d]: logfactor:%f\n", g_log_factor );
        fprintf( stderr, "[Cymatic3d]: lissscale:%f\n", g_lissajous_scale );
        fprintf( stderr, "[Cymatic3d]: lissdelay = %i\n", g_delay );
        fprintf( stderr, "[Cymatic3d]: zpos:%f\n", g_z );
        fprintf( stderr, "[Cymatic3d]: dzpos:%f\n", g_dz ); 
        fprintf( stderr, "[Cymatic3d]: spacing:%f\n", g_space );
        fprintf( stderr, "[Cymatic3d]: yview:%f\n", g_eye_y );
        fprintf( stderr, "[Cymatic3d]: depth:%i\n", g_depth );
        fprintf( stderr, "[Cymatic3d]: preview:%f (delay: %i)\n", g_wf_delay_ratio, g_wf_delay);
        fprintf( stderr, "[Cymatic3d]: rotatem:%f\n", g_inc_val_mouse );
        fprintf( stderr, "[Cymatic3d]: rotatek:%f\n", g_inc_val_kb * (INC_VAL_MOUSE/INC_VAL_KB) );
        fprintf( stderr, "[Cymatic3d]: begintime:%f (seconds)\n", g_begintime ); 
        fprintf( stderr, "[Cymatic3d]: ds:%i\n", g_ds ); 
        fprintf( stderr, "----------------------------------------------------\n" );
    break;
    }
    	
	if (!g_freezehold) g_freezehold_savestate = g_freeze;
	
        if ( key == 32 ) { // space bar
		g_freezehold = true;
		g_freeze = true;
		//fprintf( stderr, "[Cymatic3d]: freezehold ON\n"); // disabled to prevent unneeded printf's filling console
	}
}




//-----------------------------------------------------------------------------
// Name: mouseFunc( )
// Desc: handles mouse stuff
//-----------------------------------------------------------------------------
void mouseFunc( int button, int state, int x, int y )
{
  
  /// kl4yfd
  return; // No Mouse Rotation!!
  
    if( button == GLUT_LEFT_BUTTON )
    {
        // rotate
        if( state == GLUT_DOWN )
            g_y_inc -= g_inc_val_mouse;
        else
            g_y_inc += g_inc_val_mouse;
    }
    else if ( button == GLUT_RIGHT_BUTTON )
    {
        if( state == GLUT_DOWN )
            g_y_inc += g_inc_val_mouse;
        else
            g_y_inc -= g_inc_val_mouse;
    }
    else
        g_y_inc = 0.0f;

    glutPostRedisplay( );
}




//-----------------------------------------------------------------------------
// Name: idleFunc( )
// Desc: callback from GLUT
//-----------------------------------------------------------------------------
void idleFunc( )
{
    // render the scene
    glutPostRedisplay( );
}




//-----------------------------------------------------------------------------
// name: draw_string()
// desc: ...
//-----------------------------------------------------------------------------
void draw_string( GLfloat x, GLfloat y, GLfloat z, const char * str, GLfloat scale = 1.0f )
{
    GLint len = strlen( str ), i;

    glPushMatrix();
    glTranslatef( x, y, z );
    glScalef( .001f * scale, .001f * scale, .001f * scale );

    for( i = 0; i < len; i++ )
        glutStrokeCharacter( GLUT_STROKE_ROMAN, str[i] );
    
    glPopMatrix();
}




//-----------------------------------------------------------------------------
// name: ...
// desc: ...
//-----------------------------------------------------------------------------
void drawLissajous( SAMPLE * stereobuffer, int len, int channels)
{
    static float REDcolor=0.0f, GREENcolor=0.4f, BLUEcolor=1.0f ;
    static bool REDdir=1, BLUEdir=0, GREENdir=0;
    float colorSTEP = 0.01f;
    float max=0;
    static float oldmax=0.0f;
    
    static float Zdepth=0.0f;
    
    float x, y;
    static float prev1x=0, prev2x=0, prev1y=0, prev2y=0;
    SAMPLE * buffer;
    
    // 1 or 2 channels only for now
    assert( channels >= 1 && channels <= 2 );

    // mono
    if( channels == 1 )
    {
        buffer = g_cur_buffer; 
        // convert to mono
        for( int m = 0; m < len; m++)
        {
            buffer[m] = stereobuffer[m*2] + stereobuffer[m*2+1];
            buffer[m] /= 2.0f;
        }
    }
    else
    {
        buffer = stereobuffer;
    }
    
    
    
    /// kl4yfd: set thick lines
    //glLineWidth(1.25f);
    
    // color
    ///glColor3f( 1.0f, 1.0f, .5f ); // yellow
    ///glColor3f( 1.0f, 1.0f, 1.0f ); // white
    


  
    /// kl4yfd
    /// Pure random colors
    //REDcolor = static_cast<float>( (rand() % 100) / 100.0f );
    //GREENcolor = static_cast<float>( (rand() % 100) / 100.0f );
    //BLUEcolor = static_cast<float>( (rand() % 100) / 100.0f );



    
    
  ///////// Smoothly color cycle the RGB values
  /*
  /// RED
  if (REDdir)
	REDcolor -= colorSTEP;
  else 
	REDcolor += colorSTEP;
  
  if (REDcolor > 1.0f) {
	REDdir = !REDdir;
	REDcolor = 1.0f;
  } else if (REDcolor < 0.0f) {
	REDdir = !REDdir;
	REDcolor = 0.0f;
  }  
  
  /// GREEN
  if (GREENdir)
	GREENcolor += colorSTEP;
  else 
	GREENcolor -= colorSTEP;
  
  if (GREENcolor > 1.0f) {
	GREENdir = !GREENdir;
	GREENcolor = 1.0f;
  } else if (GREENcolor < 0.0f) {
	GREENdir = !GREENdir;
	GREENcolor = 0.0f;
  }  
  
  // BLUE
  if (BLUEdir)
	BLUEcolor += colorSTEP;
  else 
	BLUEcolor -= colorSTEP;
  if (BLUEcolor > 1.0f) {
	BLUEdir = !BLUEdir;
	BLUEcolor = 1.0f;
  } else if (BLUEcolor < 0.0f) {
	BLUEdir = !BLUEdir;
	BLUEcolor = 0.0f;
  }
*/
////////////  
  

/// Super experimental code !!!
/// When finished, aims to change the color of displayed cymatic image so there is a frequency-to-chakra mapping.
/// The displayed color will be that of whatever chakra the dominant-frequency is closest to. 
/*
    float dominant_freq = ((rolloff_val / SND_MARSYAS_SIZE * g_srate / 2)+(rolloff2_val / SND_MARSYAS_SIZE * g_srate / 2)) / 2;
    static const float error_margin_hz = 44;
    printf("\n %.2f", dominant_freq);

    if ( dominant_freq < 500 + error_margin_hz && dominant_freq > 500 - error_margin_hz ) {
	REDcolor = 1.0f;
	GREENcolor = 0.0f;
	BLUEcolor = 0.0f;
    } else if ( dominant_freq < 1000 + error_margin_hz && dominant_freq > 1000 - error_margin_hz ) {
	REDcolor = 0.0f;
	GREENcolor = 1.0f;
	BLUEcolor = 0.0f;
    }  else if ( dominant_freq < 1500 + error_margin_hz && dominant_freq > 1500 - error_margin_hz ) {
	REDcolor = 0.0f;
	GREENcolor = 0.0f;
	BLUEcolor = 1.0f;
    } else {
      	REDcolor = 1.0f;
	GREENcolor = 1.0f;
	BLUEcolor = 1.0f;
    }

*/
/// !!!
    /// For color-cycling
    ///glColor3f( REDcolor, GREENcolor, BLUEcolor );
    
    glColor3f(ChakraColor[chakranumber][0], ChakraColor[chakranumber][1], ChakraColor[chakranumber][2]);
    
    
    /// White
    ///glColor3f( 1,1,1 ); 
    
    /// Sets the background color
    glClearColor(ChakraColor[backgroundnumber][0], ChakraColor[backgroundnumber][1], ChakraColor[backgroundnumber][2], 1.0f);
    
    
    // save current matrix state
    glPushMatrix();
    // translate
    ///glTranslatef( 1.2f, 0.0f, 0.0f );
    // draw it
    glTranslatef( 0.0f, 0.0f, 2.0f );
    
    switch (rendertype) {
      case 0:
	    glPointSize(2.2f); glBegin( GL_POINTS );// looks awesome!
	    break;
	    
      case 1:
	    glBegin( GL_LINE_STRIP ); ///  <-- ORIGINAL!
	    break;
	    
      case 2:
	    glBegin( GL_LINES );
	    break;
	    
      case 3:
	    glBegin( GL_LINE_LOOP );
	    break;
	    
      case 4:
	    glBegin( GL_TRIANGLES );
	    break;
	    
      case 5:
	    glBegin( GL_TRIANGLE_STRIP );
	    break;
	    
      case 6:    
	    glBegin( GL_TRIANGLE_FAN ); 
	    break;
	    
      case 7:
	    glBegin( GL_QUADS );
	    break;
    
      case 8:
	    glBegin( GL_QUAD_STRIP );
	    break;      
      
      // GPU overload: overly-complex non-coherent shapes
      //case 9:
	    //glBegin( GL_POLYGON );
	    //break;    
      
    }
    
    for( int i = 0; i < len * channels; i += channels )
    {
        x = buffer[i] * g_lissajous_scale;
	
        if( channels == 1 )
        {
            // delay
            y = (i - g_delay >= 0) ? buffer[i-g_delay] : g_back_buffer[len + i-g_delay]; 
            y *= g_lissajous_scale;
        }
        else
        {
            y = buffer[i + channels-1] * g_lissajous_scale;
        }
        
        /// smoothing
        float tempx, tempy;
	
	tempx = x;
	x = (prev1x + prev2x + tempx) / 3.0f;
        prev2x = prev1x;
	prev1x = tempx;
        
	tempy = y;
	y = (prev1y + prev2y + tempy) / 3.0f;
        prev2y = prev1y;
	prev1y = tempy;

        
        
        switch(visualization)
	{
	  case 0:
	    	glVertex2f( 2.0 * x + abs(x+y), 2.0 * y + abs(x+y) );
		glVertex2f( -2.0 * x - abs(x+y), 2.0 * y + abs(x+y) );
		glVertex2f( -2.0 * x - abs(x+y), 2.0 *-y - abs(x+y) );
		glVertex2f( 2.0 * x + abs(x+y), 2.0 * -y - abs(x+y) );
		break;
		
	  case 1:
		glVertex3f(x ,sin(y) , 0.0f );
		glVertex3f(-x ,sin(y) , 0.0f );
		break;
		
	  case 2:
	    	glVertex2f( 2.0 * x, 2.0 * y );
		glVertex2f( -2.0 * x, 2.0 * y );
		glVertex2f( -2.0 * x, 2.0 *-y );
		glVertex2f( 2.0 * x, 2.0 * -y );
		break;
		
	  case 3:
		glVertex3f( x*x, y*y, 8*(x+y) );
		glVertex3f( -x*x, y*y, 8*(x+y) );
		glVertex3f( -x*x, -y*y, 8*(x+y) );
		glVertex3f( x*x, -y*y, 8*(x+y) );
		
		glVertex3f( 2*x*x, 2*y*y, 5*(x+y) );
		glVertex3f( 2*-x*x, 2*y*y, 5*(x+y) );
		glVertex3f( -2*x*x, -2*y*y, 5*(x+y) );
		glVertex3f( 2*x*x, -2*y*y, 5*(x+y) );
		
		glVertex3f( 3*x*x, 3*y*y, 3*(x+y) );
		glVertex3f( 3*-x*x, 3*y*y, 3*(x+y) );
		glVertex3f( -3*x*x, -3*y*y, 3*(x+y) );
		glVertex3f( 3*x*x, -3*y*y, 3*(x+y) );
		
		glVertex3f( 5*x*x, 5*y*y, 2*(x+y) );
		glVertex3f( 5*-x*x, 5*y*y, 2*(x+y) );
		glVertex3f( -5*x*x, -5*y*y, 2*(x+y) );
		glVertex3f( 5*x*x, -5*y*y, 2*(x+y) );
		
		glVertex3f( 8*x*x, 8*y*y, (x+y) );
		glVertex3f( 8*-x*x, 8*y*y, (x+y) );
		glVertex3f( -8*x*x, -8*y*y, (x+y) );
		glVertex3f( 8*x*x, -8*y*y, (x+y) );
		break;
  
	  case 4:
		glVertex3f( x*x, y*y, 0.0f );
		glVertex3f( -x*x, y*y, 0.0f );
		glVertex3f( x*x, -y*y, 0.0f );
		glVertex3f( -x*x, -y*y, 0.0f );
		break;

		
	  case 5:		
		glVertex3f( x*x*x, y*y*y, x+y );
		glVertex3f( -x*x*x, y*y*y, x+y );
		glVertex3f( -x*x*x, -y*y*y, x+y );
		glVertex3f( x*x*x, -y*y*y, x+y );
		break;
		
	  case 6:	     
		glVertex3f( x*x, y*y, x+y );
		glVertex3f( -x*x, y*y, x+y );
		glVertex3f( -x*x, -y*y, x+y );
		glVertex3f( x*x, -y*y, x+y );
		break;
		  
	  case 7:
		glVertex3f( x*x, y*y, x+y );
		glVertex3f( -x*x, y*y, x+y );
		glVertex3f( -x*x, -y*y, x+y );
		glVertex3f( x*x, -y*y, x+y );
		
		glVertex3f( 2*x*x, 2*y*y, 2*(x+y) );
		glVertex3f( 2*-x*x, 2*y*y, 2*(x+y) );
		glVertex3f( -2*x*x, -2*y*y, 2*(x+y) );
		glVertex3f( 2*x*x, -2*y*y, 2*(x+y) );
		
		glVertex3f( 3*x*x, 3*y*y, 3*(x+y) );
		glVertex3f( 3*-x*x, 3*y*y, 3*(x+y) );
		glVertex3f( -3*x*x, -3*y*y, 3*(x+y) );
		glVertex3f( 3*x*x, -3*y*y, 3*(x+y) );
		
		glVertex3f( 4*x*x, 4*y*y, 4*(x+y) );
		glVertex3f( 4*-x*x, 4*y*y, 4*(x+y) );
		glVertex3f( -4*x*x, -4*y*y, 4*(x+y) );
		glVertex3f( 4*x*x, -4*y*y, 4*(x+y) );
		
		glVertex3f( 5*x*x, 5*y*y, 5*(x+y) );
		glVertex3f( 5*-x*x, 5*y*y, 5*(x+y) );
		glVertex3f( -5*x*x, -5*y*y, 5*(x+y) );
		glVertex3f( 5*x*x, -5*y*y, 5*(x+y) );
		break;
	  
	  case 8:
		glVertex3f( x*x, y*y, x+y );
		glVertex3f( -x*x, y*y, x+y );
		glVertex3f( -x*x, -y*y, x+y );
		glVertex3f( x*x, -y*y, x+y );
		
		glVertex3f( 2*x*x, 2*y*y, 2*(x+y) );
		glVertex3f( 2*-x*x, 2*y*y, 2*(x+y) );
		glVertex3f( -2*x*x, -2*y*y, 2*(x+y) );
		glVertex3f( 2*x*x, -2*y*y, 2*(x+y) );
		
		glVertex3f( 3*x*x, 3*y*y, 3*(x+y) );
		glVertex3f( 3*-x*x, 3*y*y, 3*(x+y) );
		glVertex3f( -3*x*x, -3*y*y, 3*(x+y) );
		glVertex3f( 3*x*x, -3*y*y, 3*(x+y) );
		
		glVertex3f( 5*x*x, 5*y*y, 5*(x+y) );
		glVertex3f( 5*-x*x, 5*y*y, 5*(x+y) );
		glVertex3f( -5*x*x, -5*y*y, 5*(x+y) );
		glVertex3f( 5*x*x, -5*y*y, 5*(x+y) );
		
		glVertex3f( 8*x*x, 8*y*y, 8*(x+y) );
		glVertex3f( 8*-x*x, 8*y*y, 8*(x+y) );
		glVertex3f( -8*x*x, -8*y*y, 8*(x+y) );
		glVertex3f( 8*x*x, -8*y*y, 8*(x+y) );
		break;

		
	  case 9:
	    	glVertex3f( -x/2, y/2, 0.0f ); /// warp
	    	glVertex3f( x/2, -y/2, 0.0f ); /// warp
		break;
		
	  case 10:
		glVertex3f( x, y, -x*y/10 ); /// circles + sensitive burst
		glVertex3f( x, y, sqrt( x*x + y*y )/2.0f * -g_lissajous_scale ); /// circles + sensitive burst	
		break;
		
		
	  case 11:
	    	glTranslatef( 0.0f, -1.2f, 0.0f ); // vertical
	    	glVertex3f( x*x, y*y-1.0f, 0.0f ); /// vertical
	    	glVertex3f( -x*x, y*y-1.0f, 0.0f ); /// vertical
		break;
		
	  case 12:
		glVertex3f( x*x*x/10.0f, y*y*y/10.0f, 0.0f ); /// plus
		break;
	  
	  case 13:
	    	glVertex3f( x*x*x/10.0f, y*y*y/10.0f, 0.0f ); /// plus flash
	    	glVertex3f( x*x*x/-10.0f, y*y*y/-10.0f, 0.0f ); /// plus flash
		break;	  
	  
	  case 14:
		break;

	  
	  case 15: /// first 3D triangles :-)
		glVertex3f(0.0f, 0.4f, 0.0f );
	    	glVertex3f(-0.4f, -0.2f, 0.0f ); 
	    	glVertex3f(0.4f, -0.2f, 0.0f );
		
		glVertex3f(0.0f, 0.4f, -0.1f );
	    	glVertex3f(-0.4f, -0.2f, -0.1f ); 
	    	glVertex3f(0.4f, -0.2f, -0.1f ); 
		
		glVertex3f(0.0f, 0.4f, -0.2f );
	    	glVertex3f(-0.4f, -0.2f, -0.2f ); 
	    	glVertex3f(0.4f, -0.2f, -0.2f );
		
		glVertex3f(0.0f, 0.4f, -0.3f );
	    	glVertex3f(-0.4f, -0.2f, -0.3f ); 
	    	glVertex3f(0.4f, -0.2f, -0.3f ); 
		break;
		
	   case 16:
		glVertex3f( x, y, -0.04f );
		glVertex3f( x, y, -0.08f );
		glVertex3f( x, y, -0.12f );
		glVertex3f( x, y, -0.16f );
		glVertex3f( x, y, -0.20f );
		glVertex3f( x, y, -0.24f ); 
		break;
	
	   case 17:
		glVertex3f( cosf(x)-0.5, sin(y), tan(x+y) );
		glVertex3f( -cosf(x)+0.5, -sin(y), tan(x+y) );
		
		glVertex3f( -sinf(y), -cos(x)*0.5, tan(x+y) );
		glVertex3f( sinf(y), cos(x)*0.5, tan(x+y) );
		
		///glVertex3f( sinf(y), sin(x), tan(x+y) );
		break;
		
	   case 18:
		glVertex3f( x, y, abs(x+y) ); 
		break;
		
	   case 19:
		glVertex3f( x*3.0f, sinf(y)*3.0f, sqrt( x*x + y*y ) * -g_lissajous_scale ); /// tube
		glVertex3f( x, sinf(y), 0.0f ); /// tube
		break;
	   
	   case 20:
		glVertex3f( x, y, -10.0f ); /// sunburst
		glVertex3f( x, y, sqrt( x*x + y*y ) * -g_lissajous_scale ); /// sunburst
		break;
		
	  case 21:
		glVertex3f(x, y, 0.0f );
		
		glVertex3f(0.4f+x, 0.4f+y, 0.0f );
		glVertex3f(-0.4f-x, -0.4f-y, 0.0f ); 
		glVertex3f(0.4f+x, -0.4f-y, 0.0f );
		glVertex3f(-0.4f+x, 0.4f-y, 0.0f );
		
		glVertex3f(0.2f+x, 0.2f+y, 0.0f );
		glVertex3f(-0.2f-x, -0.2f-y, 0.0f ); 
		glVertex3f(0.2f+x, -0.2f-y, 0.0f );
		glVertex3f(-0.2f+x, 0.2f-y, 0.0f );
		break;
		
	  case 22:
		glVertex3f( x, y, 0.0f ); /// square / lightning
		glVertex3f( x, y, sqrt( x*x + y*y ) * -g_lissajous_scale ); /// circular
		break;
	  
	  case 23:
		glVertex3f( cosf(x), sin(y), 0.0f ); /// bowtie
		glVertex3f( -cosf(x), -sin(y), 0.0f ); /// bowtie
		break;
		
	  case 24:
		glVertex3f( tan(x), tan(y), 0.0f ); /// VERT / HORIZ lines on beat
		break;
		
	  case 25:
	    	glVertex3f( x, y, sqrt( x*x + y*y ) * -g_lissajous_scale ); /// warp disk
		glVertex3f(3* x, 3*y, -1 * sqrt( x*x + y*y ) * -g_lissajous_scale ); /// warp disk
		break;
		
	  case 26:
		glVertex3f( x*3.0f, sinf(y)*3.0f, sqrt( x*x + y*y ) * -g_lissajous_scale ); /// WOW!
		break;
		
	  case 27:
		glVertex3f( sinh(y), cosh(x)-1, 0.0f );
		glVertex3f( sinh(y), -cosh(x)+1, 0.0f );
		break;
		
	  case 28:
		break;

	  default:
		glVertex3f( x, y, 0.0f ); /// Original
		break;
	    
	}
    }
    glEnd();
    // restore matrix state
    glPopMatrix();
    
    // hmm...
    if( channels == 1 )
        memcpy( g_back_buffer, buffer, len * sizeof(SAMPLE) );
    
}




//-------------------------------------------------------------------
// Name: map_log_spacing( )
// Desc: ...
//-----------------------------------------------------------------------------
inline double map_log_spacing( double ratio, double power )
{
    // compute location
    return ::pow(ratio, power) * g_fft_size/g_freq_view; 
}




//-----------------------------------------------------------------------------
// Name: compute_log_spacing( )
// Desc: ...
//-----------------------------------------------------------------------------
double compute_log_spacing( int fft_size, double power )
{
    int maxbin = fft_size; // for future in case we want to draw smaller range
    int minbin = 0; // what about adding this one?

    for(int i = 0; i < fft_size; i++)
    {
        // compute location
        g_log_positions[i] = map_log_spacing( (double)i/fft_size, power ); 
        // normalize, 1 if maxbin == fft_size
        g_log_positions[i] /= pow((double)maxbin/fft_size, power);
    }

    return 1/::log(fft_size);
}




//-----------------------------------------------------------------------------
// Name: displayFunc( )
// Desc: callback function invoked to draw the client area
//-----------------------------------------------------------------------------
void displayFunc( )
{
    // static variables to keep across function calls
    static const int LP = 4;
    static long int count = 0;
    static char str[1024];
    ///Moved to global scope for freq-to-color mapping| static float centroid_val, flux_val, rms_val, rolloff_val, rolloff2_val;
    static fvec in(SND_MARSYAS_SIZE),
        centroid(1), flux(1), lpc(g_lpc->outSize()), mfcc(13), rms(1), rolloff(1),
        rolloff2(1), centroid_lp(LP), flux_lp(LP), rms_lp(LP), rolloff_lp(LP),
        rolloff2_lp(LP);

    // local variables
    SAMPLE * buffer = g_fft_buffer, * ptr = in.getData();
    GLfloat ytemp, fval;
    GLint i;

    // clear the color and depth buffers
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    
    // save current matrix state
    glPushMatrix( );

        // rotate the sphere about x axis
        glRotatef( g_angle_x += g_x_inc, 1.0f, 0.0f, 0.0f );
	
	// rotate the sphere about y axis
        glRotatef( g_angle_y += g_y_inc, 0.0f, 1.0f, 0.0f );
	
	// rotate the sphere about z axis
        glRotatef( g_angle_z += g_z_inc, 0.0f, 0.0f, 1.0f );
	
	// Zoom in/out in the Z (depth) axis
	glTranslatef( 0.0, 0.0, g_zoom );

        // wait for data
        while( !g_ready ) usleep( SLEEPTIME );

        // lock
        g_mutex.lock();

        // get the latest (possibly preview) window
        memset( buffer, 0, SND_FFT_SIZE * sizeof(SAMPLE) );

        // copy currently playing audio into buffer
        memcpy( buffer, g_audio_buffer, g_buffer_size * sizeof(SAMPLE) );

        // some flag (hand off to audio cb thread)
        g_ready = FALSE;

        // unlock
        g_mutex.unlock();

        // lissajous
        if( g_lissajous )
        {
            if( !g_filename ) { // real-time mic input
                drawLissajous( g_stereo_buffer, g_buffer_size, 1 );
            } else { // reading from file
                drawLissajous( g_stereo_buffer, g_buffer_size, g_sf_info.channels );
            }
        }

        // soon to be used drawing offsets
        GLfloat x = -1.8f, inc = 3.6f / g_buffer_size, y = .7f;
        // apply the transform window
        apply_window( (float*)buffer, g_window, g_buffer_size );

        // draw the time domain waveform
        if( g_waveform )
        {
            // save the current matrix state
            glPushMatrix();
            // color waveform
            glColor3f( 0.4f, 0.4f, 1.0f );
            // translate the waveform
            glTranslatef( x, y, 0.0f );
            // scale visually
            glScalef( inc * g_time_view , g_gain * g_time_scale * 2.0, 1.0 );
            // set vertex normals (for somewhat controlled lighting)
            glNormal3f( 0.0f, 0.0f, 1.0f );
            // draw waveform
            glBegin( GL_LINE_STRIP );
            {
                GLint ii = ( g_buffer_size - (g_buffer_size/g_time_view) ) / 2;
                GLfloat xcoord = 0.0f;
                // loop through samples
                for( i = ii; i < ii + g_buffer_size / g_time_view; i++ )
                {
                    glVertex2f( xcoord++ , buffer[i] );
                }
                glEnd();
            }
            // restore previous matrix state
            glPopMatrix();
        }

        // take forward FFT; result in buffer as FFT_SIZE/2 complex values
        rfft( (float *)buffer, g_fft_size/2, FFT_FORWARD );
        // cast to complex
        complex * cbuf = (complex *)buffer;

        // reset drawing offsets
        x = -1.8f;
        y = -1.0f;

        // color the spectrum
        glColor3f( 0.4f, 1.0f, 0.4f );
        // set vertex normals
        glNormal3f( 0.0f, 1.0f, 0.0f );

        // copy current magnitude spectrum into waterfall memory
        for( i = 0; i < g_fft_size/2; i++ )
        {
            // copy x coordinate
            g_spectrums[g_wf][i].x = x;
            // copy y, depending on scaling
            if( !g_usedb ) {
                g_spectrums[g_wf][i].y = g_gain * g_freq_scale * 1.8f *
                    ::pow( 25 * cmp_abs( cbuf[i] ), .5 ) + y;
            } else {
                g_spectrums[g_wf][i].y = g_gain * g_freq_scale * 
                    ( 20.0f * log10( cmp_abs(cbuf[i])/8.0 ) + 80.0f ) / 80.0f + y + .5f;
            }            
            // increment x
            x += inc * g_freq_view;
        }

        // draw the right things
        g_draw[g_wf] = g_wutrfall;
        if( !g_starting )
            g_draw[(g_wf+g_wf_delay)%g_depth] = true;
        
        // reset drawing variables
        x = -1.8f;
        inc = 3.6f / g_fft_size;
        
        // save current matrix state
        glPushMatrix();
        // translate in world coordinate
        glTranslatef( x, 0.0, g_z );
        // scale it
        glScalef( inc*g_freq_view , 1.0 , -g_space );
	
	if( g_wutrfall ) {
        // loop through each layer of waterfall
        for( i = 0; i < g_depth; i++ )
        {
            if( i == g_wf_delay || !g_freeze || g_wutrfall )
            {
                // if layer is flagged for draw
                if( g_draw[(g_wf+i)%g_depth] )
                {
                    // get the magnitude spectrum of layer
                    Pt2D * pt = g_spectrums[(g_wf+i)%g_depth];
                    // future
                    if( i < g_wf_delay )
                    {
                        // brightness based on depth
                        fval = (g_depth - g_wf_delay + i) / (float)(g_depth);
                        // rain or not
                        if( !g_rainbow ){ 
                            glColor3f( 1.0 * fval, .7 * fval, .4 * fval ); // depth cue
                            // interesting colors: (.7, 1, .2), (.4, .9. 1), (1.0, 0.7, 0.2)
                        } else {
                            // rainbow colors
                            float cval = 1 - (g_wf_delay - i) / (float)(g_wf_delay);
                            cval = 0.4f + cval * (1.0f - 0.4f);
                            glColor3f( 1.0f * fval, cval * fval, .4f * fval );
                        }
                    }
                    // present
                    else if( i == g_wf_delay )
                    {
                        // draw the now line?
                        if( g_draw_play )
                        {
                            glLineWidth( g_filename == NULL ? 2.0f : 3.0f );
                            glColor3f( .4f, 1.0f, 1.0f );
                        }
                    }
                    // past
                    else
                    {
                        // brightness based on depth
                        fval = (g_depth - i + g_wf_delay) / (float)(g_depth);
                        // draw rainbow?
                        if( !g_rainbow ) {
                            glColor3f( .4f * fval, 1.0f * fval, .4f * fval ); //depth cue
                        } else {
                            // rainbow-ish
                            float cval = 1 - (i - g_wf_delay) / (float)(g_depth - g_wf_delay);
                            cval = 0.4f + cval * (1.0f - 0.4f);
                            glColor3f( cval * fval, 1.0f * fval, .4f * fval );
                        }
                    }

                    // render the actual spectrum layer
                    glBegin( GL_LINE_STRIP );
                    for( GLint j = 0; j < g_fft_size/g_freq_view; j++, pt++ )
                    {
                        // draw the vertex
                        float d = g_backwards ? g_depth - (float) i : (float) i;
                        glVertex3f( g_log_positions[j], pt->y, d );
                    }
                    glEnd();

                    // back to default line width
                    glLineWidth(1.0f);
                }
            }
        }
	}
       
        // restore matrix state
        glPopMatrix();
        
        // if flagged, mark layer NOT to be drawn
        if( !g_wutrfall )
            g_draw[(g_wf+g_wf_delay)%g_depth] = false;

        // wtrfll
        if( !g_freeze )
        {
            // advance index
            g_wf--;
            // mod
            g_wf = (g_wf + g_depth) % g_depth; 
            // can't remember what this does anymore...
            if( g_wf == g_depth - g_wf_delay )
                g_starting = 0;
        }
        
        // The rolloff values are used for Frequency to chakra color mapping.
        g_rolloff->process( in, rolloff );
        g_rolloff2->process( in, rolloff2 );


        // calculate and draw features
        if( g_draw_features )
        {
            // if not frozen
            if( !g_freeze )
            {
                // for rough downsampling
                int ratio = g_fft_size / SND_MARSYAS_SIZE / 2;
                // get magnitude response
                for( i = 0; i < SND_MARSYAS_SIZE; i++ )
                    ptr[i] = cmp_abs( cbuf[i*ratio] );
        
                // centroid
                g_centroid->process( in, centroid );
                // flux
                g_flux->process( in, flux );
                // rms
                g_rms->process( in, rms );
                
		//rolloff 1
		// Moved to a higher-scope for frequency to color mapping
                //g_rolloff->process( in, rolloff );
		// rolloff 2
                //g_rolloff2->process( in, rolloff2 );
        
                // lowpass
                centroid_lp(count % LP) = centroid(0);
                flux_lp(count % LP) = flux(0);
                rms_lp(count % LP) = rms(0);
                rolloff_lp(count % LP) = rolloff(0);
                rolloff2_lp(count % LP) = rolloff2(0);
                count++;

                // get average values
                centroid_val = centroid_lp.mean();
                flux_val = flux_lp.mean();
                rms_val = rms_lp.mean();
                rolloff_val = rolloff_lp.mean();
                rolloff2_val = rolloff2_lp.mean();
            }

            // draw the centroid
            // TODO: need to update 'inc'?
            ytemp = y+.04f + 2 * (::pow( 30 * rms_val, .5 ) );
            float centroid_x = map_log_spacing( centroid_val/SND_MARSYAS_SIZE, g_log_factor );
            centroid_x *= (inc*g_freq_view);
            glColor3f( 1.0f, .4f, .4f );
            glBegin( GL_LINE_STRIP );
              glVertex3f( -1.8f + centroid_x, ytemp, 0.0f + g_z );
              glVertex3f( -1.8f + centroid_x, y-.04f, 0.0f + g_z );
            glEnd();
    
            // centroid value
            glBegin( GL_LINE_STRIP );
              glVertex3f( -1.8f + centroid_x, y-.04f, 0.0f + g_z );
              glVertex3f( -1.72f + centroid_x, y-.15f, 0.0f + g_z );
              glVertex3f( -1.15f + centroid_x, y-.15f, 0.0f + g_z );
            glEnd();
        
            // print centroid
            sprintf( str, "centroid = %.0f Hz", centroid_val / 2 / SND_MARSYAS_SIZE * g_srate / 2 );
            draw_string( -1.7f + centroid_x, y-.14f, 0.0f + g_z, str, .4f );

            // rms value
            glBegin( GL_LINE_STRIP );
              glVertex3f( -1.8f + centroid_x - .23f, ytemp, 0.0f + g_z );
              glVertex3f( -1.8f + centroid_x + .23f, ytemp, 0.0f + g_z );
            glEnd();
            
            // print RMS
            sprintf( str, "RMS = %f", 1000 * rms_val );
            draw_string( -1.8f + centroid_x - .23f, ytemp + .01f, 0.0f + g_z, str, 0.4f );
            
            // draw the rolloff
            glColor3f( 1.0f, 1.0f, .4f );
            float rolloff_x = map_log_spacing( rolloff_val/SND_MARSYAS_SIZE, g_log_factor );
            rolloff_x *= (inc*g_freq_view);
            glBegin( GL_LINE_STRIP );
              glVertex3f( -1.8f + rolloff_x, y-.04f, 0.0f + g_z );
              glVertex3f( -1.8f + rolloff_x, y+.04f, 0.0f + g_z );
            glEnd();

            // draw other rolloff
            glColor3f( 1.0f, 1.0f, 1.0f );
            float rolloff2_x = map_log_spacing( rolloff2_val/SND_MARSYAS_SIZE, g_log_factor );
            rolloff2_x *= (inc*g_freq_view);
            glBegin( GL_LINE_STRIP );
              glVertex3f( -1.8f + rolloff2_x, y-.04f, 0.0f + g_z );
              glVertex3f( -1.8f + rolloff2_x, y+.04f, 0.0f + g_z );
            glEnd();
        
            // centroid
            sprintf( str, "centroid = %.0f", centroid_val / SND_MARSYAS_SIZE * g_srate / 2 );
            draw_string( -1.7f, 0.4f, 0.0f, str, 0.4f );
            // flux
            sprintf( str, "flux = %.1f", flux_val );
            draw_string( -1.7f, 0.3f, 0.0f, str, 0.4f );
            // flux
            sprintf( str, "RMS = %.4f", 1000 * rms_val );
            draw_string( -1.7f, 0.2f, 0.0f, str, 0.4f );
            // flux
            sprintf( str, "50%% rolloff= %.0f", rolloff_val / SND_MARSYAS_SIZE * g_srate / 2 );
            draw_string( -1.7f, 0.1f, 0.0f, str, 0.4f );
            // flux
            sprintf( str, "80%% rolloff = %.0f", rolloff2_val / SND_MARSYAS_SIZE * g_srate / 2 );
            draw_string( -1.7f, 0.0f, 0.0f, str, 0.4f );
        }

        // print to console
        if( g_stdout )
        {
            fprintf( stdout, "%.2f  %.2f  %.8f  %.2f  %.2f  ", centroid(0), flux(0), rms(0), rolloff(0), rolloff2(0) );
            fprintf( stdout, "%.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f %.2f %.2f  ", 
                     mfcc(0), mfcc(1), mfcc(2), mfcc(3), mfcc(4), mfcc(5), mfcc(6),
                     mfcc(7), mfcc(8), mfcc(9), mfcc(10), mfcc(11), mfcc(12) );
            fprintf( stdout, "\n" );
        }

        // set color
        glColor3f( 1, 1, 1 );

        // title
        // draw_string( 0.0f, 0.0f, 0.0f, "Cymatic3d + watrfall", .5f );

        // time
        if( g_show_time )
        {
            float fsec = ((double)sf_seek(g_sf,0,SEEK_CUR) - (double)g_wf_delay * g_buffer_size) / (double)g_srate;
            sprintf( str, "%.0f", fsec ); 
            draw_string( -1.7f, 1.1f, -.2f, str, .4f );
        }

        // pause?
        if( g_pause )
            ///draw_string( 0.95f, 1.1f, -.2f, "paused... (press f to resume)", .4f );

        // mute?
        if( g_mute )
            draw_string( 0.95f, 1.05f, -.2f, "muted... (press m to unmute)", .4f );
	
	/// Pause for a few frames to better see details (eye-time)
	/// only while NOT holding the space bar to freeze
        if (g_freezecycle && !g_freezehold) {
		static unsigned int freezecounter = 0;
		if (freezecounter++ < freezecycle_skipframes) {
			g_freeze = true;
		} else {
			g_freeze = false;
			freezecounter = 0;
		}
	}

    // restore matrix state
    glPopMatrix( );

    // flush gl commands
    glFlush( );
    // swap the buffers
    glutSwapBuffers( );

    // maintain count from render
    g_buffer_count_b++;
    // check against count from reading function
    if( g_filename && !g_file_running && g_buffer_count_b == g_buffer_count_a )
        g_running = FALSE;
}




//-----------------------------------------------------------------------------
// Name: extract_buffer( )
// Desc: extract one buffer
//-----------------------------------------------------------------------------
void extract_buffer( )
{
    // static stuff
    static fvec raw(g_buffer_size), in(SND_MARSYAS_SIZE),
        centroid(1), flux(1), lpc(g_lpc->outSize()), mfcc(13), rms(1), rolloff(1),
        rolloff2(1);
    
    // local
    SAMPLE buffer[SND_BUFFER_SIZE], * ptr = in.getData();
    GLint i;

    // wait for reading    
    while( !g_ready )
        usleep( SLEEPTIME );

    // get data
    if( !g_filename )
    {
        g_mutex.lock();
        memcpy( buffer, g_audio_buffer, g_buffer_size * sizeof(SAMPLE) );
        g_ready = FALSE;
        g_mutex.unlock();
    }
    else
    {
        memcpy( buffer, g_audio_buffer, g_buffer_size * sizeof(SAMPLE) );
        g_ready = FALSE;
    }

    // apply the window
    apply_window( (float*)buffer, g_window, g_buffer_size );

    // take the forward fft, leaving fftsize/2 complex values
    rfft( (float *)buffer, g_buffer_size/2, FFT_FORWARD );
    // cast to complex
    complex * cbuf = (complex *)buffer;

    // get magnitude spectrum
    for( i = 0; i < g_buffer_size/2; i++ )
        ptr[i] = cmp_abs( cbuf[i] );

    // centroid
    g_centroid->process( in, centroid );
    // flux
    g_flux->process( in, flux );
    // lpc
    g_lpc->process( in, lpc );
    // mfcc
    g_mfcc->process( in, mfcc );
    // rms
    g_rms->process( in, rms );
    // rolloff
    g_rolloff->process( in, rolloff );
    g_rolloff2->process( in, rolloff2 );

    // print to console
    if( g_stdout )
    {
        fprintf( stdout, "%.2f\t%.2f\t%.8f\t%.0f\t%.2f  ", centroid(0), flux(0), rms(0), rolloff(0)*rolloff(0), rolloff2(0) );
        fprintf( stdout, "%.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f %.2f %.2f  ", 
                 mfcc(0), mfcc(1), mfcc(2), mfcc(3), mfcc(4), mfcc(5), mfcc(6),
                 mfcc(7), mfcc(8), mfcc(9), mfcc(10), mfcc(11), mfcc(12) );
        fprintf( stdout, "\n" );
    }

    // file reading stuff
    g_buffer_count_b++;
    if( g_filename && !g_file_running && g_buffer_count_a == g_buffer_count_b )
        g_running = FALSE;
}
