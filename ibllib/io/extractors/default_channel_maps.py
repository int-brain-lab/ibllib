DEFAULT_MAPS = {
    'ephys':
        {'3A':
            {'ap':
                {'left_camera': 2,
                 'right_camera': 3,
                 'body_camera': 4,
                 'bpod': 7,
                 'frame2ttl': 12,
                 'rotary_encoder_0': 13,
                 'rotary_encoder_1': 14,
                 'audio': 15
                 }
             },
         '3B':
            {'nidq':
                {'left_camera': 0,
                 'right_camera': 1,
                 'body_camera': 2,
                 'imec_sync': 3,
                 'frame2ttl': 4,
                 'rotary_encoder_0': 5,
                 'rotary_encoder_1': 6,
                 'audio': 7,
                 'bpod': 16,
                 'laser': 17,
                 'laser_ttl': 18},
             'ap':
                {'imec_sync': 6}
             },
         },

    'widefield':
        {'nidq': {'left_camera': 0,
                  'right_camera': 1,
                  'body_camera': 2,
                  'frame_trigger': 3,
                  'frame2ttl': 4,
                  'rotary_encoder_0': 5,
                  'rotary_encoder_1': 6,
                  'audio': 7,
                  'bpod': 16}
         },

    'sync':
        {'nidq': {'left_camera': 0,
                  'right_camera': 1,
                  'body_camera': 2,
                  'frame_trigger': 3,
                  'frame2ttl': 4,
                  'rotary_encoder_0': 5,
                  'rotary_encoder_1': 6,
                  'audio': 7,
                  'bpod': 16}
         }
}
