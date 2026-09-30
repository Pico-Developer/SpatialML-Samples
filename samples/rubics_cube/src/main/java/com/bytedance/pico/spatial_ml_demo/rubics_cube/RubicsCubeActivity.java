package com.bytedance.pico.spatial_ml_demo.rubics_cube;

import android.Manifest;
import android.app.NativeActivity;
import android.content.pm.PackageManager;
import android.os.Bundle;
public class RubicsCubeActivity extends NativeActivity {

    static {
        System.loadLibrary("rubics_cube");
    }

    private static final int REQ_CAMERA = 1001;

    public void requestCameraFromNative() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.CAMERA}, REQ_CAMERA);
        } else {
//            nativeOnPermissionResult(REQ_CAMERA,
//                    new String[]{Manifest.permission.CAMERA},
//                    new int[]{PackageManager.PERMISSION_GRANTED});
        }
    }

    @Override
    public void onRequestPermissionsResult(int rc, String[] perms, int[] grants) {
        super.onRequestPermissionsResult(rc, perms, grants);
//        nativeOnPermissionResult(rc, perms, grants);
    }

//    private static native void nativeOnPermissionResult(int rc, String[] perms, int[] grants);
}
