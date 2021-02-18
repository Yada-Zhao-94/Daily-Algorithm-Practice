# Daily-Algorithm-Practice
[02-06-2021: 给定 100G 的 URL 磁盘数据，使用最多 1G 内存，统计出现频率最高的 Top K 个 URL](#02-06-2021给定-100g-的-url-磁盘数据使用最多-1g-内存统计出现频率最高的-top-k-个-url)  
[02-07-2021: 10亿个数中如何高效地找到最大的一个数以及最大的第K个数](#02-07-2021-10亿个数中如何高效地找到最大的一个数以及最大的第k个数)  
[02-07-2021: 合并两个有序链表](#02-07-2021合并两个有序链表-leetcode)  
[02-07-2021: 64匹马，8个赛道，找出前4匹马最少需要比几次](#02-07-202164匹马8个赛道找出前4匹马最少需要比几次)  
[02-07-2021: 实现快速排序](#02-07-2021-实现快速排序)  
[02-08-2021: 两数相加 II (Leetcode 445)](#02-08-2021445-两数相加-ii)  
[02-08-2021: 二叉树的锯齿形层次遍历 (Leetcode)](#02-08-2021二叉树的锯齿形层次遍历)  
[02-09-2021: K个一组翻转链表 (Leetcode 25)](#02-09-2021-k-个一组翻转链表-leetcode-25)  
[02-10-2021: 搜索旋转排序数组 (Leetcode 33)](#02-10-2021-搜索旋转排序数组-leetcode-33)  
[02-11-2021: 判断有环链表的环长度](#02-11-2021-判断有环链表的环长度)  
[02-12-2021: 寻找旋转排序数组中的最小值(数组有重复/无重复元素)]()  
[02-13-2021: 最大子序和(Leetcode 53)]()  
[02-14-2021: 给定一个 foo 函数，60%的概率返回0，40%的概率返回1，如何利用 foo 函数实现一个 50% 返回 0 的函数？(以及利用均匀硬币产生不等概率)]()  

## 02-06-2021: 给定 100G 的 URL 磁盘数据，使用最多 1G 内存，统计出现频率最高的 Top K 个 URL
1. 新建约100个文件，利用hash(URL) % 100的值，将每条URL映射到对应文件下，保证同一URL必然全部映射到同一文件下。
2. 假定分布比较均匀，则每个文件大小约为1G，可在内存中操作，在内存中简单地统计词频即可。
3. 对每个文件都取出相应topK的K个<URL，count>根据count降序排列，形成类似一个数组，将100个文件的数组写入同一个文件。
4. 对100个K对进行[K路归并](https://leetcode-cn.com/problems/merge-k-sorted-lists/). 得到K个即可  
*即使100个K对不能都放进内存：取每个数组的一段放入内存，在K路归并步骤中，在这一段已完全被Heap吐出时，再向堆中插入这一段的下一个<URL, count>即可。
## 02-07-2021: 10亿个数中如何高效地找到最大的一个数以及最大的第K个数  
基本同上。假设一个数为几十个byte大小，10亿个数为几十GB级别，不能全部放入内存。  
**1）最基础解法：K路归并**
1. 一次处理内存放得下的数据量，得到最大的K个数（PriorityQueue或quick selection法）
2. 对所有最大的K个数的数组进行K路归并

**2）询问数据是否有取值或分布范围，则可以使用桶排序/计数排序**  
*3）bash自带的sort可排序超过内存大小的文件 
## 02-07-2021：合并两个有序链表 (Leetcode)
[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)
简单
## 02-07-2021：64匹马，8个赛道，找出前4匹马最少需要比几次
智力题，11次   
https://blog.csdn.net/star_fighting/article/details/104706155/
## 02-07-2021: 实现快速排序
Partition -> 利用递归，Sort左，Sort右
```Java
public class Solution {
  public int[] quickSort(int[] array) {
    // Write your solution here
    if (array == null || array.length == 0) {
      return array;
    }
    quickSort(array, 0, array.length - 1);
    return array;
  }

  private void quickSort(int[] array, int lo, int hi) {
    if (lo >= hi) {
      return;
    }
    int pivot = partition(array, lo, hi);
    quickSort(array, lo, pivot - 1);
    quickSort(array, pivot + 1, hi);
  }

  private int partition(int[] array, int lo, int hi) {
    int i = lo + 1, j = hi;
    int val = array[lo];
    while(i <= j) {
      if(array[i] <= val) {
        i++;
      } else if(array[j] >= val) {
        j--;
      } else {
        swap(array, i, j);
        i++;
        j--;
      }
    }
    swap(array, lo, j);
    return j;
  }

  private void swap(int[] array, int i, int j) {
    int temp = array[j];
    array[j] = array[i];
    array[i] = temp;
  }
}
```

## 02-08-2021：445. 两数相加 II
[leetcode 445](https://leetcode-cn.com/problems/add-two-numbers-ii/)  
直接将两链表转化为数字再相加会导致数字溢出❌  
还是采用两数相加I的末位开始处理，最后反转链表。
```Java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Deque<Integer> stack1 = new LinkedList<>();
        Deque<Integer> stack2 = new LinkedList<>();
        while(l1 != null) {
            stack1.push(l1.val);
            l1 = l1.next;
        }
        while(l2 != null) {
            stack2.push(l2.val);
            l2 = l2.next;
        }
        int add1 = 0;
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while(!stack1.isEmpty() || !stack2.isEmpty()) {
            int num1 = stack1.isEmpty() ? 0 : stack1.pop();
            int num2 = stack2.isEmpty() ? 0 : stack2.pop();
            int res = num1 + num2 + add1;
            ListNode node = new ListNode(res % 10);
            cur.next = node;
            cur = node;
            add1 = res / 10;
        }
        if (add1 == 1) {
            cur.next = new ListNode(1);
        }
        ListNode newHead = reverse(dummy.next);
        return newHead;
    }

    private ListNode reverse(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = reverse(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
}
```

## 02-08-2021：二叉树的锯齿形层次遍历
[leetcode 103](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)  
使用Deque,在BFS过程中两头倒腾即可
```Java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offerFirst(root);
        int count = 1;
        while(!deque.isEmpty()) {
            int size = deque.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < size; i++) {               
                if (count % 2 != 0) {
                    TreeNode cur = deque.pollFirst();
                    list.add(cur.val);
                    if (cur.left != null) {
                        deque.offerLast(cur.left);
                    }
                    if (cur.right != null) {
                        deque.offerLast(cur.right);
                    }
                } else {
                    TreeNode cur = deque.pollLast();
                    list.add(cur.val);
                    if (cur.right != null) {
                        deque.offerFirst(cur.right);
                    }
                    if (cur.left != null) {
                        deque.offerFirst(cur.left);
                    }
                }
            }
            count++;
            res.add(list);
        }
        return res;
    }
}
```
## 02-09-2021: K 个一组翻转链表 (Leetcode 25)
由于只能使用常数的额外空间，不能使用recursion. 所以只能用类似iterative反转链表的方法。  
prev为已处理好的前半部分，当前k个Node需要反转，next为后半段。注意while循环中，需要最后将cur设置为已被反转k个Node这段的Tail，从而进入下一循环
```Java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode cur = dummy;
        while(true) {
            ListNode prev = cur;
            for(int i = 0; i < k; i++) {
                cur = cur.next;
                if (cur == null) {
                    return dummy.next;
                }
            }
            ListNode next = cur.next;
            ListNode oldHead = prev.next;
            cur.next = null;
            ListNode[] res = reverse(oldHead);
            prev.next = res[0];
            res[1].next = next;
            cur = res[1];
        }
    }

    private ListNode[] reverse(ListNode head) {
        ListNode[] res = new ListNode[2];
        res[1] = head;
        ListNode prev = null;
        while(head != null) {
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        res[0] = prev;
        return res;
    }
}
```
## 02-10-2021: 搜索旋转排序数组 (Leetcode 33)
观察：经过旋转后的数组，任意取一个区间，左半段右半段中至少有一个是递增的。再根据target是否在这半段取值内，即可实现搜索区间减半
```Java
class Solution {
    public int search(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while(lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] >= nums[lo]) {
                if (target >= nums[lo] && target <= nums[mid]) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            } else {
                if (target >= nums[mid] && target <= nums[hi]) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
        return -1;
    }
}
```
## 02-11-2021: 判断有环链表的环长度
判断有环：快慢指针，假设slow指针走了n步的话。可推导出当 n = k * (环长度) 时，两指针都会相遇一次。  
第一次相遇后，第二个指针再走一圈即可得环长度
```Java
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                int cycleLength = getLength(fast, slow);
                System.out.println(cycleLength);
                return true;
            }
        }
        return false;
    }

    private int getLength(ListNode fast, ListNode slow) {
        int count = 0;
        do {
            fast = fast.next;
            count++;
        } while(fast != slow);
        return count;
    }
}
```
## 02-12-2021: 寻找旋转排序数组中的最小值(数组有重复/无重复元素) Leetcode 153, 154
数组无重复元素：153   
递归法，分别求左右两半的最小值。值得注意的是**如何证明复杂度为O(logN)：** T(N) = T(N/2) + O(1) -> 是因为每次对半分区间后，至少有一半是排序数组。
```Java
class Solution {
    public int findMin(int[] nums) {
        return findMin(nums, 0, nums.length - 1);
    }

    private int findMin(int[] nums, int lo, int hi) {
        // first base case: only 2 or 1 elements
        if (lo + 1 >= hi) {
            return Math.min(nums[lo], nums[hi]);
        }
        // second base case: one half must be sorted
        if (nums[lo] < nums[hi]) {
            return nums[lo];
        }
        int mid = lo + (hi - lo) / 2;
        return Math.min(findMin(nums, lo, mid), findMin(nums, mid + 1, hi));
    }
}
```
数组有重复元素：154    
与153代码一样，但我们要注意当nums[lo] == nums[hi]时，不能推断出lo ~ hi是排序的。例子：33413  
所以时间复杂度为O(N)
```Java
class Solution {
    public int findMin(int[] nums) {
        return findMin(nums, 0, nums.length - 1);
    }

    private int findMin(int[] nums, int lo, int hi) {
        // first base case: only 2 or 1 elements
        if (lo + 1 >= hi) {
            return Math.min(nums[lo], nums[hi]);
        }
        // second base case: if we can find a half is sorted
        // if (nums[lo] == nums[hi]), we can't conclude that [lo ~ hi] is sorted
        // 33313
        if (nums[lo] < nums[hi]) {
            return nums[lo];
        }
        int mid = lo + (hi - lo) / 2;
        return Math.min(findMin(nums, lo, mid), findMin(nums, mid + 1, hi));
    }
}
```

## 02-13-2021: 最大子序和(Leetcode 53)
设dp[i]为0~i最大连续子数组和，且必须包含nums[i] 
```Java
class Solution {
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int max = nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for(int i = 1; i < nums.length; i++) {
            if (dp[i - 1] > 0) {
                dp[i] = dp[i - 1] + nums[i];
            } else {
                dp[i] = nums[i];
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }
}
```
## 02-14-2021: 给定一个 foo 函数，60%的概率返回0，40%的概率返回1，如何利用 foo 函数实现一个 50% 返回 0 的函数？
分析连续抛出两次硬币的情况，正反面的出现有四种情况，概率依次为：  
(1) 两次均为正面：0.6 * 0.6=0.36  
(2)第一次正面，第二次反面：0.6 * 0.4=0.24  
(3)第一次反面，第二次正面：0.4 * 0.6=0.24  
(4)两次均为反面：0.4 * 0.4=0.16  
可以看到中间两种情况的概率是完全一样的，于是问题的解法就是连续抛两次硬币，如果两次得到的相同则重新抛两次；否则根据第一次（或第二次）的正面反面情况，就可以得到两个概率相等的事件。

```Java
public int coin() {
  while(true) {
    int a = foo();
    int b = foo();
    if (a != b) {
      return a;
    }
  }
 }
```
***
利用均匀硬币产生不等概率  
问题描述：有一枚均匀的硬币，抛出此硬币后，可用foo()表示其结果。已知foo()能返回0和1两个值，其概率均为0.5。问怎么利用foo()得到另一个函数，使得返回0和1的概率分别为0.3和0.7。  
问题分析：0和1随机生成，可以理解为二进制。可以令a=foo()*2^3+foo()*2^2+foo()*2^1+foo()等概率生成0-15的所有数，只取1~10之间的数，则产生1到10之间的数仍为等概率。%3为0时返回0，%3不为0时返回1
```Java
private int generateNum() {
  while(true) {
    int a = foo()*2^3 + foo()*2^2 + foo()*2^1 + foo();
    if (a >=1 && a <= 10) {
      return a;
    }
  }    
}

public int coin() {
  int random = generateNum();
  if (random % 3 == 0) {
    return 0;
  } else {
    return 1;
  }
}
```


