package com.example.user.liv_icon.Network.Task;

import android.graphics.Bitmap;
import android.os.AsyncTask;

import com.example.user.liv_icon.Network.HttpRequest;

import java.util.HashMap;
import java.util.Map;

public class ImageRequestTask extends AsyncTask<String, Integer, String> {

    private ImageRequestTaskHandler handler;

    public interface ImageRequestTaskHandler{
        public void onSuccessTask(String result);
        public void onFailTask();
        public void onCancelTask();
    }

    public ImageRequestTask(ImageRequestTaskHandler handler){
        this.handler = handler;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
    }

    @Override
    protected String doInBackground(String... strings) {
        String url = strings[0];
        String path = strings[1];

        Map<String,Object> params = new HashMap<String,Object>();

        if(strings[2] != null){
            params.put("img",strings[2]);
        }
        params.put("key",String.valueOf(System.currentTimeMillis()));

        String result = null;

        HttpRequest request = new HttpRequest();

        try{
            String str = request.callRequestServer(url,path,"POST",params);

            result = str;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return result;
    }


    @Override
    protected void onPostExecute(String result) {
        super.onPostExecute(result);

        if(result != null){
            handler.onSuccessTask(result);
        }
    }
}
