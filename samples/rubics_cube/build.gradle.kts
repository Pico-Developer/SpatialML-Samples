plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    compileSdk = 34
    ndkVersion = "26.3.11579264"
    namespace = "com.bytedance.pico.secure_mr_demo.rubics_cube"
    defaultConfig {
        minSdk = 34
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        applicationId = "com.bytedance.pico.secure_mr_demo.rubics_cube"
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
        getByName("debug") {
            isDebuggable = true
            isJniDebuggable = true
        }
        getByName("release") {
            isDebuggable = false
            isJniDebuggable = false
        }
    }
    flavorDimensions += "version"
    externalNativeBuild {
        cmake {
            version = "3.22.1"
            path("CMakeLists.txt")
        }
    }
    sourceSets {
        getByName("main") {
            manifest.srcFile("AndroidManifest.xml")
            assets.srcDirs("../../assets/rubics_cube")
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
