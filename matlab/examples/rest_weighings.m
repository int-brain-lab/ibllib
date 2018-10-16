one = One();
%%

subject = '437';
wei = one.alyx_client.get(['/weighings?nickname=' subject]);

% wei = one.alyx_client.get(['/weighings']);
wei.date_time = time.jsonrest2serial(wei.date_time);
[~,ordre] = sort(wei.date_time );
wei = sel_struct(wei, ordre);


%%
figure,plot(wei.date_time, wei.weight,'.')


%%
wa = one.alyx_client.get(['/water-administrations']);
wa = one.alyx_client.get(['/water-administrations?nickname=' subject]);
wa.date_time = time.jsonrest2serial(wa.date_time);
