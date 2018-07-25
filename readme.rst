Description
===========

TemporalMedian is a temporal denoising filter. It replaces every pixel with
the median of its temporal neighbourhood.

This filter will introduce ghosting, so use with caution.


Usage
=====
::

    tmedian.TemporalMedian(clip clip[, int radius=1, int[] planes=all])


Parameters:
    *clip*
        A clip to process. It must have constant format and dimensions
        and it must be 8..16 bit with integer samples or 32 bit with
        float samples.

    *radius*
        Size of the temporal neighbourhood. Must be between 1 and 10.
        
        The first and the last *radius* frames of the clip are
        returned unchanged.

        Default: 1.

    *planes*
        Planes to filter. Planes that aren't filtered will be copied
        from the input.

        Default: all.


Compilation
===========

::

    mkdir build && cd build
    meson ../
    ninja


License
=======

ISC.
