#ifndef PIPELINE_H
#define PIPELINE_H

#include <vector>
#include <string>
#include <initializer_list>

class PipelineElement;

class Pipeline {
public:
    Pipeline(std::initializer_list<PipelineElement*> elems);
    void run();
private:
    std::vector<PipelineElement* > elements;
};

#endif
