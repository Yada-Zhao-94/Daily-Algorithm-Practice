# Daily-Algorithm-Practice
[02-06-2021: 给定 100G 的 URL 磁盘数据，使用最多 1G 内存，统计出现频率最高的 Top K 个 URL](#02-06-2021给定-100g-的-url-磁盘数据使用最多-1g-内存统计出现频率最高的-top-k-个-url)  
[02-07-2021: 10亿个数中如何高效地找到最大的一个数以及最大的第K个数](#02-07-2021-10亿个数中如何高效地找到最大的一个数以及最大的第k个数)  
[02-07-2021: 合并两个有序链表](#02-07-2021合并两个有序链表-leetcode)  
[02-07-2021: 64匹马，8个赛道，找出前4匹马最少需要比几次](#02-07-202164匹马8个赛道找出前4匹马最少需要比几次)  
[02-07-2021: 实现快速排序](#02-07-2021-实现快速排序)  
[02-08-2021: 两数相加 II (Leetcode 445)](#02-08-2021445-两数相加-ii)  
[02-08-2021: 二叉树的锯齿形层次遍历 (Leetcode)](#02-08-2021二叉树的锯齿形层次遍历)  
[02-09-2021: K个一组翻转链表 (Leetcode 25)](#02-09-2021-k-个一组翻转链表-leetcode-25)  
[02-09-2021: 搜索旋转排序数组 (Leetcode 33)](#02-09-2021-搜索旋转排序数组-leetcode-33)  

## 02-06-2021:给定 100G 的 URL 磁盘数据，使用最多 1G 内存，统计出现频率最高的 Top K 个 URL
1. 新建约100个文件，利用hash(URL) % 100的值，将每条URL映射到对应文件下，保证同一URL必然全部映射到同一文件下。
2. 假定分布比较均匀，则每个文件大小约为1G，可在内存中操作，在内存中简单地统计词频即可。
3. 对每个文件都取出相应topK的K个<URL，count>根据count降序排列，形成类似一个数组，将100个文件的数组写入同一个文件。
4. 对100个K对进行[K路归并](https://leetcode-cn.com/problems/merge-k-sorted-lists/). 得到K个即可  
*即使100个K对不能都放进内存：取每个数组的一段放入内存，在K路归并步骤中，在这一段已完全被Heap吐出时，再向堆中插入这一段的下一个<URL, count>即可。
## 02-07-2021: 10亿个数中如何高效地找到最大的一个数以及最大的第K个数  
基本同上。假设一个数为几十个byte大小，10亿个数为几十GB级别，不能全部放入内存。  
**1）最基础解法：K路归并**
1. 一次处理内存放得下的数据量，得到最大的K个数（PriorityQueue或qucik selection法）
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

## 02-09-2021: 搜索旋转排序数组 (Leetcode 33)
