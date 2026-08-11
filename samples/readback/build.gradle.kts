plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    compileSdk = 34
    ndkVersion = "26.3.11579264"
    namespace = "com.bytedance.pico.secure_mr_demo.readback"
    defaultConfig {
        minSdk = 34
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        applicationId = "com.bytedance.pico.secure_mr_demo.readback"
        externalNativeBuild {
            cmake {
                arguments.add("-DANDROID_STL=c++_shared")
                arguments.add("-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=OFF")
            }
            ndk {
                abiFilters.add("arm64-v8a")
            }
        }
    }
    lint {
        disable.add("ExpiredTargetSdkVersion")
    }
    buildFeatures {
        prefab = true
        viewBinding = true
    }
    buildTypes {
        debug {
            isDebuggable = true
            isJniDebuggable = true
        }
        release {
            isDebuggable = false
            isJniDebuggable = false
        }
    }
    flavorDimensions += "version"
    productFlavors {
        create("readback_cpu") {
            dimension = "version"

        }
        create("readback_vulkan") {
            dimension = "version"
            externalNativeBuild {
                cmake {
                    arguments.add("-DREADBACK_USE_GPU=1")
                    arguments.add("-DREADBACK_USE_VULKAN=1")
                }
            }
        }
        create("readback_opengl") {
            dimension = "version"
            externalNativeBuild {
                cmake {
                    arguments.add("-DREADBACK_USE_GPU=1")
                    arguments.add("-DREADBACK_USE_OPENGL=1")
                }
            }
        }
    }
    externalNativeBuild {
        cmake {
            version = "3.22.1"
            path("CMakeLists.txt")
        }
    }
    sourceSets {
        getByName("main") {
            manifest.srcFile("AndroidManifest.xml")
            assets.srcDirs("../../assets/common", "../../assets/pose")
            java.srcDirs("src/main/java")
        }
    }
    packaging {
        jniLibs {
            keepDebugSymbols.add("**.so")
        }
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
}
dependencies {
//    implementation("com.android.support:appcompat-v7:28.0.0")
//    implementation("androidx.activity:activity-ktx:1.11.0")
    implementation(project(":spatialml-xr-utils"))
    implementation("androidx.appcompat:appcompat:1.7.0")
}
