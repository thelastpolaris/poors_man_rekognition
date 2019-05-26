#ifndef PIPELINE_ELEMENT_H
#define PIPELINE_ELEMENT_H

class Pipeline;

class PipelineElement {
public:
    PipelineElement();
    virtual void run() =0;
private:
    Pipeline* parentPipeline;
};

#endif
