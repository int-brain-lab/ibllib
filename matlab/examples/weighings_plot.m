one = One('alyx_login','root','alyx_pwd','Pradesh71','alyx_url','https://alyx.internationalbrainlab.org');
%%

subject = '437';
wei = one.alyx_client.get(['/weighings?nickname=' subject]);
wei.date_time = time.jsonrest2serial(wei.date_time);
[~,ordre] = sort(wei.date_time );
wei = sel_struct(wei, ordre);


%%
figure,plot(wei.date_time, wei.weight)


