package com.example.user.liv_icon.Filter;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.os.AsyncTask;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Base64;
import android.view.View;
import android.widget.ImageView;

import com.example.user.liv_icon.R;
import com.zomato.photofilters.geometry.Point;
import com.zomato.photofilters.imageprocessors.Filter;
import com.zomato.photofilters.imageprocessors.subfilters.BrightnessSubfilter;
import com.zomato.photofilters.imageprocessors.subfilters.ColorOverlaySubfilter;
import com.zomato.photofilters.imageprocessors.subfilters.ContrastSubfilter;
import com.zomato.photofilters.imageprocessors.subfilters.SaturationSubfilter;
import com.zomato.photofilters.imageprocessors.subfilters.ToneCurveSubfilter;

import java.util.ArrayList;

public class FilterActivity extends AppCompatActivity {
    static
    {
        System.loadLibrary("NativeImageProcessor");
    }

    ImageView imageView;
    RecyclerView recyclerView;
    Filter_adapter filter_adapter;
    public static Bitmap img1,img2,img3,img4,img5;
    SharedPreferences mPref;
    Bitmap decodedBitmap;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_filter);
        imageView = (ImageView)findViewById(R.id.image);
        recyclerView = (RecyclerView)findViewById(R.id.list);
        filter_adapter = new Filter_adapter(this);

        mPref = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
        String base_img = mPref.getString("img", "0");

        byte[] decodedByteArray = Base64.decode(base_img, Base64.NO_WRAP);
        decodedBitmap = BitmapFactory.decodeByteArray(decodedByteArray, 0, decodedByteArray.length);
        decodedBitmap =decodedBitmap.copy(Bitmap.Config.ARGB_8888, true);

        imageView.setImageBitmap(decodedBitmap);


        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        linearLayoutManager.setOrientation(LinearLayoutManager.HORIZONTAL);
        recyclerView.setLayoutManager(linearLayoutManager);

        recyclerView.setAdapter(filter_adapter);

        recyclerView.addOnItemTouchListener(new RecyclerItemClickListener(getApplicationContext(), recyclerView, new RecyclerItemClickListener.OnItemClickListener() {
            @Override
            public void onItemClick(View view, int position) {
                if(position ==0){
                    ToneCurveSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==1){
                    ColorOverlaySubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==2){
                    ContrastSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==3){
                    BrightnessSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==4){
                    SaturationSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
            }

            @Override
            public void onLongItemClick(View view, int position) {
                if(position ==0){
                    ToneCurveSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==1){
                    ColorOverlaySubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==2){
                    ContrastSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==3){
                    BrightnessSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==4){
                    SaturationSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
            }
        }));

        img1=  BitmapFactory.decodeResource(getResources(), R.drawable.dokyo);
        img1 =img1.copy(Bitmap.Config.ARGB_8888, true);

        img2=  BitmapFactory.decodeResource(getResources(), R.drawable.dokyo);
        img2 =img2.copy(Bitmap.Config.ARGB_8888, true);

        img3=  BitmapFactory.decodeResource(getResources(), R.drawable.dokyo);
        img3 =img3.copy(Bitmap.Config.ARGB_8888, true);

        img4=  BitmapFactory.decodeResource(getResources(), R.drawable.dokyo);
        img4 =img4.copy(Bitmap.Config.ARGB_8888, true);

        img5=  BitmapFactory.decodeResource(getResources(), R.drawable.dokyo);
        img5 =img5.copy(Bitmap.Config.ARGB_8888, true);

        ToneCurveSubfilter(img1);
        ColorOverlaySubfilter(img2);
        ContrastSubfilter(img3);
        BrightnessSubfilter(img4);
        SaturationSubfilter(img5);
    }




    public void ToneCurveSubfilter(Bitmap bitmap){
        Filter myFilter = new Filter();

        Point[] rgbKnots;
        rgbKnots = new Point[3];
        rgbKnots[0] = new Point(0, 0);
        rgbKnots[1] = new Point(105, 139);
        rgbKnots[2] = new Point(255, 255);

        myFilter.addSubFilter(new ToneCurveSubfilter(rgbKnots, null, null, null));
        img1 = myFilter.processFilter(bitmap);
    }

    public void ColorOverlaySubfilter(Bitmap bitmap){
        Filter myFilter = new Filter();

        myFilter.addSubFilter(new ColorOverlaySubfilter(100, .3f, .4f, .0f));
        img2 = myFilter.processFilter(bitmap);
    }

    public void ContrastSubfilter(Bitmap bitmap){
        Filter myFilter = new Filter();
        myFilter.addSubFilter(new ContrastSubfilter(2.2f));
        img3 = myFilter.processFilter(bitmap);
    }

    public void BrightnessSubfilter(Bitmap bitmap){
        Filter myFilter = new Filter();
        myFilter.addSubFilter(new BrightnessSubfilter(60));
        img4 = myFilter.processFilter(bitmap);
    }

    public void SaturationSubfilter(Bitmap bitmap){
        Filter myFilter = new Filter();
        myFilter.addSubFilter(new SaturationSubfilter(1.3f));
        img5 = myFilter.processFilter(bitmap);
    }

}
