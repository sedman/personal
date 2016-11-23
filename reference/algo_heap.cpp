#include <queue>
#include <iostream>
#include <cassert>
#include <vector>
#include <climits>

template<typename T, int N>
class heap
{
 private:
  std::size_t idx; // empty pointer
  T list[N + 1];
  void swap(T &data1, T &data2)
  {
    T tmp = data1;
    data1 = data2;
    data2 = tmp;
  }

 public:
  heap()
  :idx(1) 
  {
  }

  void push(T data)
  {
    if(idx > N)
      throw std::logic_error("out of bound");
    int child_idx = idx;
    list[idx] = data;
    int parent_idx = idx / 2;
    while(parent_idx >= 1 && list[child_idx] > list[parent_idx])
    {
      swap(list[child_idx], list[parent_idx]); 
      child_idx = parent_idx;
      parent_idx /= 2;
    }
    ++idx;
  }

  std::size_t size() const { return idx - 1; }
  bool empty(void) const { return idx == 1; }

  T top(void) 
  {
    if(size() < 1)
      throw std::logic_error("no data");
    if(size() == 1)
    {
      --idx;
      return list[1];
    }
    int result = list[1];
    --idx;
    swap(list[idx], list[1]);
    int parent_idx = 1;
    int child_idx = 2; 
    while(child_idx < idx)
    {
      int bigger_child_idx = child_idx;
      if(child_idx + 1 < idx && list[child_idx] < list[child_idx + 1])
        bigger_child_idx = child_idx + 1;
      if(list[parent_idx] < list[bigger_child_idx])
      {
        swap(list[parent_idx], list[bigger_child_idx]);
        parent_idx = bigger_child_idx;
        child_idx = parent_idx * 2;
      }
      else
        break;
    }
    return result; 
  }  
};

void assert_list(std::vector<int> &testset)
{
  heap<int, 100> my_heap;
  std::priority_queue<int> std_heap;

  for(int data : testset)
  {
    my_heap.push(data);
    std_heap.push(data);
  }

  for(int idx = 0; idx < testset.size(); ++idx)
  {
    assert(my_heap.top() == std_heap.top());
    std_heap.pop();
  }
}

int main(void)
{
  std::vector<int> test1 { 3, 7, 7, 10, 1, -10, 11, 100000, 1, 2030483, -1000, INT_MAX };
  assert_list(test1);

  std::vector<int> test2 { -1, -1, -1, -1, -1, -1, -1, -1};
  assert_list(test2);

  std::cout<<"passed\n";
  return 0;
}
