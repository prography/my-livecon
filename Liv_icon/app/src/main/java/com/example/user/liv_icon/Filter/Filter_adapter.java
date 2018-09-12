package com.example.user.liv_icon.Filter;

import android.content.Context;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.example.user.liv_icon.R;

import static com.example.user.liv_icon.Filter.FilterActivity.img1;
import static com.example.user.liv_icon.Filter.FilterActivity.img2;
import static com.example.user.liv_icon.Filter.FilterActivity.img3;
import static com.example.user.liv_icon.Filter.FilterActivity.img4;
import static com.example.user.liv_icon.Filter.FilterActivity.img5;


public class Filter_adapter extends RecyclerView.Adapter<Filter_holder> {

    Context context;

    public Filter_adapter(Context context){
        this.context = context;
    }

    @Override
    public Filter_holder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.activity_holder,parent,false);

        return new Filter_holder(view);
    }

    @Override
    public void onBindViewHolder(Filter_holder holder, int position) {

        if(position ==0){
            holder.setFilter(img1,"Normal");
        }
        else if(position ==1){
            holder.setFilter(img2,"Natural");
        }

        else if(position ==2){
            holder.setFilter(img3,"Clarendon");
        }

        else if(position ==3){
            holder.setFilter(img4,"Lark");
        }

        else if(position ==4){
            holder.setFilter(img5,"PurPlus");
        }



    }

    @Override
    public int getItemCount() {
        return 5;
    }
}
