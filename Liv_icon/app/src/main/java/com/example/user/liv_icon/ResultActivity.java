package com.example.user.liv_icon;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.user.liv_icon.Network.Task.ImageRequestTask;

import java.util.Random;

public class ResultActivity extends AppCompatActivity {
    SharedPreferences mPref;
    Bitmap decodedBitmap;
    ImageView imageView;
    String base_img = "";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        imageView = (ImageView)findViewById(R.id.result);

        mPref = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
        base_img = mPref.getString("img", "0");


        ImageRequestTask requestTask = new ImageRequestTask(new ImageRequestTask.ImageRequestTaskHandler() {
            @Override
            public void onSuccessTask(String result) {

                byte[] decodedByteArray = Base64.decode(result, Base64.NO_WRAP);
                decodedBitmap = BitmapFactory.decodeByteArray(decodedByteArray, 0, decodedByteArray.length);
                decodedBitmap =decodedBitmap.copy(Bitmap.Config.ARGB_8888, true);

                imageView.setImageBitmap(decodedBitmap);
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


        requestTask.execute("http://52.79.148.129/", "upload", base_img);

    }


}
