import { app } from "../../scripts/app.js";

// Preserve named settings across frontend versions that serialize converted
// STRING/INT widgets differently. Never let a shifted boolean become a COMBO.
app.registerExtension({
  name:"BFS.H3.NativeNamedInputs",
  async beforeRegisterNodeDef(type,data){
    if(data.name!=="BFSH3NativeReferenceSetToVideo")return;
    const configure=type.prototype.onConfigure;
    type.prototype.onConfigure=function(info){
      const result=configure?.apply(this,arguments);
      const saved=info.properties?.bfs_h3_native_inputs;
      const values=Array.isArray(info.widgets_values)?info.widgets_values:[];
      const enumIndex=values.findIndex(v=>v==="match"||v==="max");
      const inferred=enumIndex>=3?{
        width:values[enumIndex-3],height:values[enumIndex-2],length:values[enumIndex-1],
        ref_image_size:values[enumIndex],validate_h3_sections:values[enumIndex+1],parse_prompt_weights:values[enumIndex+2]
      }:{};
      const settings=saved||inferred;
      for(const widget of this.widgets||[]){
        const value=settings[widget.name];
        if(widget.name==="ref_image_size")widget.value=["match","max"].includes(value)?value:["match","max"].includes(widget.value)?widget.value:"match";
        else if(["width","height","length"].includes(widget.name)&&Number.isInteger(value)&&value>0)widget.value=value;
        else if(["validate_h3_sections","parse_prompt_weights"].includes(widget.name)&&typeof value==="boolean")widget.value=value;
      }
      return result;
    };
    const serialize=type.prototype.onSerialize;
    type.prototype.onSerialize=function(info){
      const result=serialize?.apply(this,arguments);
      info.properties??={};info.properties.bfs_h3_native_inputs={};
      for(const widget of this.widgets||[])if(["width","height","length","ref_image_size","validate_h3_sections","parse_prompt_weights"].includes(widget.name))info.properties.bfs_h3_native_inputs[widget.name]=widget.value;
      return result;
    };
  }
});
