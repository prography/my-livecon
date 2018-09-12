package com.example.user.liv_icon;

import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;

import com.example.user.liv_icon.Filter.FilterActivity;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

public class MainActivity  extends AppCompatActivity implements View.OnClickListener {

    public static final int PICK_CAMERA = 101;

    String base_img;
    public static final int PICK_ALBUM = 102;
    Bitmap bitmap = null;
    private Uri mImageCaptureUri;
    private ImageButton inputImg;
    private boolean isSelectImage = false;
    SharedPreferences mPref;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        inputImg = (ImageButton)findViewById(R.id.inputImg);

        mPref = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

        inputImg.setOnClickListener(this);
    }

    //카메라에서 이미지 가져오기
    private void doTakePhotoAction(){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        // 임시 캐시폴더에 이미지를 저장하기 위한 작업
        File photo = new File(getExternalCacheDir(), "Meitu_" + System.currentTimeMillis() + ".png");

        try {
            // 파일 생성
            photo.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }

                /*
                버전에 따라 file uri를 접근하는 방식이 다르다.
                특히 7.0 이상에서는 권한 문제로 인해 함부로 접근이 불가하여 xml 폴더에 있는 provider에 먼저 정의 후
                접근 할 수 있도록 작업해야함
                */
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            mImageCaptureUri= FileProvider.getUriForFile(this, BuildConfig.APPLICATION_ID + ".provider", photo);
        } else {
            mImageCaptureUri = Uri.fromFile(photo);
        }

        intent.putExtra(android.provider.MediaStore.EXTRA_OUTPUT, mImageCaptureUri);
        startActivityForResult(intent, PICK_CAMERA);
    }

    //앨범에서 이미지 가져오는 것
    private void doTakeAlbumAction(){
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType(MediaStore.Images.Media.CONTENT_TYPE);
        intent.setData(MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_ALBUM);
    }

    public String getBase64String(Bitmap bitmap)
    {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);

        byte[] imageBytes = byteArrayOutputStream.toByteArray();

        return Base64.encodeToString(imageBytes, Base64.NO_WRAP);
    }

    private void bitmapImage(Uri uri){


        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        try {
            BitmapFactory.decodeStream(getContentResolver().openInputStream(uri), null, options);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        try {
            bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri), null, options);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        inputImg.setImageBitmap(bitmap);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        switch (requestCode){
            case PICK_CAMERA:

                bitmapImage(mImageCaptureUri);
                base_img = getBase64String(bitmap);

                SharedPreferences.Editor editor = mPref.edit();

                editor.putString("img", base_img);

                editor.commit();

                Intent intent = new Intent(MainActivity.this, FilterActivity.class);
                startActivity(intent);
                break;

            case PICK_ALBUM:

                if(data == null){
                    break;
                }else{
                    Uri mImageCaptureUri = data.getData();
                    bitmapImage(mImageCaptureUri);
                    base_img = getBase64String(bitmap);

                    SharedPreferences.Editor editor1 = mPref.edit();

                    editor1.putString("img", base_img);

                    editor1.commit();

                    Intent intent1 = new Intent(MainActivity.this, FilterActivity.class);
                    startActivity(intent1);

                }
                break;

        }
    }


    public void onClick(View v) {

        switch (v.getId()) {

            case R.id.inputImg:

                { DialogInterface.OnClickListener cameraListener = new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            doTakePhotoAction();
                        }
                    };

                    DialogInterface.OnClickListener albumListener = new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            doTakeAlbumAction();
                        }
                    };

                    DialogInterface.OnClickListener cancelListener = new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            dialog.dismiss();
                        }
                    };

                    new AlertDialog.Builder(this)
                            .setTitle("업로드할 이미지 선택")
                            .setPositiveButton("앨범선택", albumListener)
                            .setNeutralButton("사진촬영", cameraListener)
                            .setNegativeButton("취소", cancelListener)
                            .show();
                }
                break;

        }
    }


    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {

    }
}
