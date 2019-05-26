#include "pipeline.h"
#include "pipeline_elements/pipeline_element.h"
#include <iostream>

Pipeline::Pipeline(std::initializer_list<PipelineElement*> elems)
    :elements(elems)
{
    std::cout << elements.size() << '\n';
}

void Pipeline::run() {
    for(auto const& val: elements) {
        val->run();
    }
}
