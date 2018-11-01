package com.example.user.liv_icon;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.user.liv_icon.Network.Task.ImageRequestTask;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Member;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Random;

public class ResultActivity extends AppCompatActivity {
    SharedPreferences mPref;
    Bitmap decodedBitmap;
    ImageView imageView;
    String image = "";
    String converImg = "";
    String key = "";
    String firstkey = "";
    Button save;
    Gson gson = new Gson();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        imageView = (ImageView)findViewById(R.id.result);
        save = findViewById(R.id.save);

        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                save();

                Toast.makeText(getApplicationContext(), "저장이 완료되었습니다.", Toast.LENGTH_SHORT).show();


//
            }
        });


        mPref = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
        image = mPref.getString("img", "0");
        firstkey = mPref.getString("key", "0");


        ImageRequestTask requestTask = new ImageRequestTask(new ImageRequestTask.ImageRequestTaskHandler() {
            @Override
            public void onSuccessTask(String result) {
                JsonObject jsonObject = new Gson().fromJson(result, JsonObject.class);

                key = jsonObject.get("key").getAsString();

                if(key.equals(firstkey)){
                    converImg = jsonObject.get("img").getAsString();
                    byte[] decodedString = Base64.decode(converImg, Base64.DEFAULT);
                    decodedBitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);

                    imageView.setImageBitmap(decodedBitmap);
                }



            }

            @Override
            public void onFailTask() {
                Toast.makeText(getApplicationContext(),"서버에서 불러오는데 실패하였습니다.",Toast.LENGTH_LONG);
            }

            @Override
            public void onCancelTask() {
                Toast.makeText(getApplicationContext(),"사용자가 해당 작업을 중지하였습니다.",Toast.LENGTH_LONG);
            }

        });


        requestTask.execute("http://35.231.246.9/", "imageUpload", image, firstkey);

    }

    public void save() {
        String filename;
        Date date = new Date(0);
        SimpleDateFormat sdf = new SimpleDateFormat ("yyyyMMddHHmmss");
        filename =  sdf.format(date);

        try{
            String path = Environment.getExternalStorageDirectory().toString();
            OutputStream fOut = null;
            File file = new File(path, "/DCIM/Screenshots/"+filename+".jpg");
            fOut = new FileOutputStream(file);

            decodedBitmap.compress(Bitmap.CompressFormat.JPEG, 85, fOut);
            fOut.flush();
            fOut.close();

            MediaStore.Images.Media.insertImage(getContentResolver()
                    ,file.getAbsolutePath(),file.getName(),file.getName());

        }catch (Exception e) {
            e.printStackTrace();
        }

    }


}
