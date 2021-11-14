<?php
function combinations($arrays, $i = 0) {
    if (!isset($arrays[$i])) {
        return array();
    }
    if ($i == count($arrays) - 1) {
        return $arrays[$i];
    }

    // get combinations from subsequent arrays
    $tmp = combinations($arrays, $i + 1);

    $result = array();

    // concat each array from tmp with each element from $arrays[$i]
    foreach ($arrays[$i] as $v) {
        foreach ($tmp as $t) {
            $result[] = is_array($t) ?
                array_merge(array($v), $t) :
                array($v, $t);
        }
    }

    return $result;
}
$arrs = combinations(
    array(
        array('xanh','vang','do'),
        array('xanh','vang','do'),
        array('xanh','vang','do')
    )
);
$arr_color = ['xanh' => "#46d832", 'vang' => "#f0dc45", "do" => "#e43b3b"];
$arr_new = [];
foreach($arrs as $arr){
  $color = [];
  foreach($arr as $type){
    $color[] = $arr_color[$type];
  }
  $name = implode('_', $arr);
  $arr_new[] = ['name' => $name, 'colors' => $color];
}
echo json_encode($arr_new);
?>
