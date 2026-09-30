package com.bytedance.pico.spatial_ml_demo.mnistwild;

import android.Manifest;
import android.app.NativeActivity;
import android.content.pm.PackageManager;

public class MnistWildActivity extends NativeActivity {

    static {
        System.loadLibrary("mnistwild");
    }

    private static final int REQ_CAMERA = 1001;

    public void requestCameraFromNative() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        } else {
//            nativeOnPermissionResult(REQ_CAMERA,
//                    new String[]{Manifest.permission.CAMERA},
//                    new int[]{PackageManager.PERMISSION_GRANTED});
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
//        nativeOnPermissionResult(requestCode, permissions, grantResults);
    }

//    private static native void nativeOnPermissionResult(int rc, String[] perms, int[] grants);
}
