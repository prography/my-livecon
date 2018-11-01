package com.example.user.liv_icon.Filter;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.user.liv_icon.R;
import com.example.user.liv_icon.ResultActivity;
import com.zomato.photofilters.geometry.Point;
import com.zomato.photofilters.imageprocessors.Filter;
import com.zomato.photofilters.imageprocessors.subfilters.ColorOverlaySubfilter;
import com.zomato.photofilters.imageprocessors.subfilters.ContrastSubfilter;
import com.zomato.photofilters.imageprocessors.subfilters.ToneCurveSubfilter;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.concurrent.Future;

import retrofit2.Response;

import static com.example.user.liv_icon.MainActivity.getBase64String;


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
    String img;
    Bitmap decodedBitmap;
    Button upload;
    String base_str;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_filter);
        imageView = (ImageView)findViewById(R.id.image);
        recyclerView = (RecyclerView)findViewById(R.id.list);
        filter_adapter = new Filter_adapter(this);
        upload = (Button)findViewById(R.id.upload);

        setImage();

        imageView.setImageBitmap(decodedBitmap);


        upload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                base_str = getBase64String(decodedBitmap);

                SharedPreferences.Editor editor = mPref.edit();

                editor.putString("img", base_str);

                editor.putString("key",String.valueOf(System.currentTimeMillis()));

                editor.commit();

                Intent intent = new Intent(FilterActivity.this, ResultActivity.class);
                startActivity(intent);
            }
        });

        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        linearLayoutManager.setOrientation(LinearLayoutManager.HORIZONTAL);
        recyclerView.setLayoutManager(linearLayoutManager);

        recyclerView.setAdapter(filter_adapter);

        recyclerView.addOnItemTouchListener(new RecyclerItemClickListener(getApplicationContext(), recyclerView, new RecyclerItemClickListener.OnItemClickListener() {
            @Override
            public void onItemClick(View view, int position) {
                if(position ==0){
                    setImage();
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==1){
                    setImage();

                    decodedBitmap = ToneCurveSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==2){
                    setImage();
                    decodedBitmap = ContrastSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==3){
                    setImage();
                    decodedBitmap = BrightnessSubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
                else if(position ==4){
                    setImage();
                    decodedBitmap = ColorOverlaySubfilter(decodedBitmap);
                    imageView.setImageBitmap(decodedBitmap);
                }
            }

            @Override
            public void onLongItemClick(View view, int position) {
            }
        }));


        init();
        img2  = ToneCurveSubfilter(img1);
        init();
        img3 = ContrastSubfilter(img1);
        init();
        img4 = BrightnessSubfilter(img1);
        init();
        img5 = ColorOverlaySubfilter(img1);
        init();


    }

    public void init(){
        img1=  BitmapFactory.decodeResource(getResources(), R.drawable.dokyo);
        img1 =img1.copy(Bitmap.Config.ARGB_8888, true);

    }


    public void setImage(){
        mPref = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
        String base_img = mPref.getString("image", "0");

        byte[] decodedByteArray = Base64.decode(base_img, Base64.NO_WRAP);
        decodedBitmap = BitmapFactory.decodeByteArray(decodedByteArray, 0, decodedByteArray.length);
        decodedBitmap =decodedBitmap.copy(Bitmap.Config.ARGB_8888, true);
    }

    public Bitmap ToneCurveSubfilter(Bitmap bitmap){
        Filter myFilter = new Filter();

        Point[] rgbKnots;
        rgbKnots = new Point[3];
        rgbKnots[0] = new Point(0, 0);
        rgbKnots[1] = new Point(105, 139);
        rgbKnots[2] = new Point(255, 255);

        myFilter.addSubFilter(new ToneCurveSubfilter(rgbKnots, null, null, null));
        img2 = myFilter.processFilter(bitmap);
        return img2;

    }

    public Bitmap ColorOverlaySubfilter(Bitmap bitmap){
        Filter myFilter = new Filter();
        myFilter.addSubFilter(new ColorOverlaySubfilter(100, .5f, .2f, .5f));
        img5 = myFilter.processFilter(bitmap);
        return img5;

    }

    public Bitmap ContrastSubfilter(Bitmap bitmap){
        Filter myFilter = new Filter();
        myFilter.addSubFilter(new ContrastSubfilter(2.2f));
        img3 = myFilter.processFilter(bitmap);
        return img3;

    }

    public Bitmap BrightnessSubfilter(Bitmap bitmap){
        Filter myFilter = new Filter();
        myFilter.addSubFilter(new ColorOverlaySubfilter(100, .3f, .6f, .6f));
        img4 = myFilter.processFilter(bitmap);
        return img4;

    }




}
