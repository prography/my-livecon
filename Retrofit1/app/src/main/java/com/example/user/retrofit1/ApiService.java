package com.example.user.retrofit1;

import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Query;

public interface ApiService {
    public static final String API_URL = "http://jsonplaceholder.typicode.com/";

    //즉 http://jsonplaceholder.typicode.com/comments?postId=1을 의미

    @GET("comments")
    Call<ResponseBody>getComment(@Query("postId")int postId);

}
