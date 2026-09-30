package com.bytedance.pico.spatial_ml_demo.readback;

import android.Manifest;
import android.app.NativeActivity;
import android.content.pm.PackageManager;
import android.os.Bundle;

public class ReadbackActivity extends NativeActivity {
    static {
        System.loadLibrary("readback");
    }

    private static final int REQ_CAMERA = 1001;
    /**
     * This Activity requires the camera permission because the sample accesses
     * camera input for VST image readback purposes.
     *
     * The SPATIAL_DATA permission is not required for this sample, but may be
     * required for other readback use cases.
     */
    public void requestCameraFromNative() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.CAMERA}, REQ_CAMERA);
        } 
        else
        {
            nativeSetPermission(android.Manifest.permission.CAMERA, true);
        }
    }

    @Override
    public void onRequestPermissionsResult(int rc, String[] perms, int[] grants) {
        super.onRequestPermissionsResult(rc, perms, grants);
        for (int i = 0; i < perms.length; i++)
        {
            if (perms[i].equals(android.Manifest.permission.CAMERA))
                nativeSetPermission(perms[i], grants[i] == PackageManager.PERMISSION_GRANTED);
        }
    }

    public native void nativeSetPermission(String permission, boolean granted);
}
