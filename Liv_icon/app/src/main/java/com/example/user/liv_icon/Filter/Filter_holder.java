package com.example.user.liv_icon.Filter;

import android.graphics.Bitmap;
import android.support.v7.widget.RecyclerView;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.user.liv_icon.R;

public class Filter_holder extends RecyclerView.ViewHolder {
        ImageView imageView;
        TextView textView;

                public Filter_holder(View itemView) {
                        super(itemView);
                        imageView = (ImageView)itemView.findViewById(R.id.filter_img);
                        textView = (TextView)itemView.findViewById(R.id.filter_name);
                        }

                public void setFilter(Bitmap img, String str){
                        imageView.setImageBitmap(img);
                        textView.setText(str);
                        }
        }
