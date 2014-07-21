

var Predictor = function(opts) {
  this.init();
  this.framecount=0;
  this.lastframe = new convnetjs.Vol(256,240,1,0.0);
};

(function() {
  
  Predictor.prototype = {
    init: function() {
      
      var layer_defs = [];
      layer_defs.push({type:'input', out_sx:256, out_sy:240, out_depth:1});
      layer_defs.push({type:'split'});
      layer_defs.push({type:'conv', sx:5, filters:2, stride:1, pad:2, activation:'relu'});  //8
      layer_defs.push({type:'pool', sx:4, stride:4});
      layer_defs.push({type:'split'});
      layer_defs.push({type:'conv', sx:5, filters:2, stride:1, pad:2, activation:'relu'});  //8
      layer_defs.push({type:'pool', sx:50, stride:50});
      layer_defs.push({type:'split'});
      layer_defs.push({type:'corner', sx:256, sy:240});
      layer_defs.push({type:'fc', num_neurons:4, activation:'relu', preserve_sx: 1, preserve_sy: 1});
      layer_defs.push({type:'fc', num_neurons:4, activation:'relu', preserve_sx: 1, preserve_sy: 1});
      layer_defs.push({type:'fc', num_neurons:4, activation:'relu', preserve_sx: 1, preserve_sy: 1});
      layer_defs.push({type:'fc', num_neurons:1, preserve_sx: 1, preserve_sy: 1});
      layer_defs.push({type:'regression'});

      this.net = new convnetjs.Net();
      this.net.makeLayers(layer_defs);

      this.trainer = new convnetjs.SGDTrainer(this.net, {method:'adadelta', batch_size:1, l2_decay:0.001});

    },
    ingest: function(oldbuffer, buffer, keys) {
      if (++this.framecount % 60) return;
      
      var output = new convnetjs.Vol(256,240,1,0.0);
      for (var i=0; i<256*240; i++) output.w[i] = (buffer[i] && 0xFF)/0xFF;  // red channel only

      if (!this.lastframe) {
        this.lastframe = output;
        return;
      }
      
      var stats = this.trainer.train(this.lastframe, output.w);
      this.lastframe = output;
      
      $('#prediction')[0].innerHTML='';
      draw_activations($('#prediction')[0], this.net.layers.slice(-1)[0].out_act);
      
      $('#stats').text(JSON.stringify(stats));
      
      if (this.framecount % 600) return;
      visualize_activations(this.net, $('#visual')[0]);
    }
  }
}());
